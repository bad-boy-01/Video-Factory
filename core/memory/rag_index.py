"""
Novel RAG Index
===============
Lightweight semantic retrieval over previously-processed novel chunks.

The problem this solves: StoryBibleGeneratorStage and SceneSplitterStage both
process a novel chunk by chunk. Without this, the only continuity mechanism
between chunk N and something established back in chunk 3 is whatever made it
into the STRUCTURED fields already extracted (character appearance, location
tags). A callback to a specific object, a line of dialogue, or a foreshadowed
detail that never got a structured field is simply gone by the time a much
later chunk needs it. This index lets a stage pull back the actual prose of
the most relevant earlier chunks before extracting/planning, instead of
relying only on structured summaries of what came before.

Design choices:
  - Brute-force cosine similarity. A full novel chunked at ~500-2000 chars
    rarely exceeds a few hundred chunks, so there's no need for an
    approximate-nearest-neighbor library (faiss, etc.) at this scale.
  - Uses sentence-transformers if available, and degrades to a complete no-op
    (never raises) if it isn't installed - this is enrichment, not a required
    dependency, and the pipeline must run without it exactly as before.
  - chunk_id is expected to be a content hash (as ChunkerStage already
    produces), so re-adding the same chunk_id is treated as a no-op rather
    than an update - the same hash can only correspond to the same text.
  - Persisted to disk under the project workspace so a Kaggle crash/restart
    doesn't lose already-computed embeddings.
"""

import os
import json
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class NovelRAGIndex:
    MODEL_NAME = "all-MiniLM-L6-v2"  # ~80MB, CPU-friendly, sufficient for this
    MAX_EMBED_CHARS = 2000  # truncate long chunks - the opening captures the gist

    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.embeddings_path = os.path.join(self.cache_dir, "embeddings.npy")
        self.chunks_path = os.path.join(self.cache_dir, "chunks.json")

        self._model = None
        self._available: Optional[bool] = None  # tri-state: None = not checked yet

        self._chunk_ids: List[str] = []
        self._chunk_texts: List[str] = []
        self._embeddings = None  # numpy array, shape (N, dim), or None

        self._load()

    # ── Availability ──────────────────────────────────────────────────────────

    def _ensure_model(self) -> bool:
        """Lazily loads the embedding model. Returns True if ready to use."""
        if self._available is not None:
            return self._available
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.MODEL_NAME)
            self._available = True
            logger.info(f"[RAG] Loaded embedding model '{self.MODEL_NAME}'.")
        except ImportError:
            logger.warning(
                "[RAG] 'sentence-transformers' not installed - chunk retrieval "
                "disabled for this run (pipeline continues without it). "
                "Install with: pip install sentence-transformers"
            )
            self._available = False
        except Exception as e:
            logger.warning(f"[RAG] Failed to load embedding model ({e}) - retrieval disabled.")
            self._available = False
        return self._available

    def is_available(self) -> bool:
        return self._ensure_model()

    # ── Adding chunks ─────────────────────────────────────────────────────────

    def add_chunk(self, chunk_id: str, text: str):
        """
        Embed and store a chunk for future retrieval. Safe to call even if the
        embedding model isn't available (becomes a no-op), and safe to call
        more than once with the same chunk_id (skipped - chunk_id is a content
        hash, so a repeat can only mean the same text).
        """
        if chunk_id in self._chunk_ids:
            return
        if not self._ensure_model():
            return

        import numpy as np

        vec = self._embed(text)
        self._chunk_ids.append(chunk_id)
        self._chunk_texts.append(text)
        if self._embeddings is None:
            self._embeddings = vec.reshape(1, -1)
        else:
            self._embeddings = np.vstack([self._embeddings, vec.reshape(1, -1)])
        self._save()

    def _embed(self, text: str):
        return self._model.encode(text[: self.MAX_EMBED_CHARS], normalize_embeddings=True)

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query_text: str,
        before_chunk_id: Optional[str] = None,
        top_k: int = 3,
    ) -> List[Dict]:
        """
        Return the top_k most semantically similar chunks to query_text,
        restricted to chunks added strictly BEFORE before_chunk_id (if given),
        so a stage never retrieves content from later in the story than the
        chunk it's currently processing.

        Returns [] if retrieval isn't available or there's nothing indexed yet
        - callers should treat this as optional enrichment, never a required
        input, and must produce a normal prompt either way.
        """
        if not self._ensure_model() or self._embeddings is None or not self._chunk_ids:
            return []

        import numpy as np

        if before_chunk_id is not None and before_chunk_id in self._chunk_ids:
            cutoff = self._chunk_ids.index(before_chunk_id)
        else:
            cutoff = len(self._chunk_ids)

        if cutoff == 0:
            return []

        query_vec = self._embed(query_text)
        candidates = self._embeddings[:cutoff]
        scores = candidates @ query_vec  # cosine similarity (both sides normalized)

        k = min(top_k, cutoff)
        top_indices = np.argsort(scores)[::-1][:k]

        return [
            {
                "chunk_id": self._chunk_ids[i],
                "text": self._chunk_texts[i],
                "similarity": float(scores[i]),
            }
            for i in top_indices
        ]

    def format_context(self, results: List[Dict], max_chars_each: int = 400) -> str:
        """Formats retrieve() results into a prompt-ready block. Returns '' if empty."""
        if not results:
            return ""
        parts = []
        for r in results:
            excerpt = r["text"][:max_chars_each]
            if len(r["text"]) > max_chars_each:
                excerpt += "..."
            parts.append(f"- {excerpt}")
        return "Relevant earlier context from this story:\n" + "\n".join(parts)

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self):
        try:
            with open(self.chunks_path, "w", encoding="utf-8") as f:
                json.dump({"chunk_ids": self._chunk_ids, "chunk_texts": self._chunk_texts}, f)
            if self._embeddings is not None:
                import numpy as np
                np.save(self.embeddings_path, self._embeddings)
        except Exception as e:
            logger.warning(f"[RAG] Failed to persist index: {e}")

    def _load(self):
        if os.path.exists(self.chunks_path):
            try:
                with open(self.chunks_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._chunk_ids = data.get("chunk_ids", [])
                self._chunk_texts = data.get("chunk_texts", [])
            except Exception as e:
                logger.warning(f"[RAG] Failed to load chunk index: {e}")

        if os.path.exists(self.embeddings_path):
            try:
                import numpy as np
                self._embeddings = np.load(self.embeddings_path)
            except Exception as e:
                logger.warning(f"[RAG] Failed to load embeddings: {e}")
