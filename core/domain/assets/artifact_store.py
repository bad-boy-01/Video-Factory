"""
ArtifactStore — Content-Addressable Storage for all pipeline artifacts.

Replaces the unused AssetRegistry with a properly functioning store that:
  • Is populated by every generation step (images, audio, video clips)
  • Implements the Image Bank: content-hash based reuse
  • Tracks full provenance (generator, seed, prompt hash, QA scores)
  • Enables incremental rebuilds (check if artifact exists before regenerating)
  • Supports thumbnail generation for inspection

File layout:
    workspace/
        artifact_store.json      ← index
        outputs/                 ← rendered images/clips (unchanged)
        audio/                   ← voiceover WAVs (unchanged)

Format: artifacts are indexed by content_hash for CAS semantics,
and also indexed by shot_id for fast pipeline lookup.
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ArtifactEntry(BaseModel):
    """Metadata record for one generated artifact."""

    artifact_id: str
    """Unique ID, typically = content_hash[:16]"""

    artifact_type: str
    """'image' | 'audio' | 'video_clip' | 'final_video' | 'manifest'"""

    path: str
    """Absolute or workspace-relative path to the file."""

    content_hash: str
    """SHA-256 of the file's bytes. Used as the CAS key."""

    # Provenance
    shot_id: str = ""
    scene_id: str = ""
    generator: str = ""
    """Model or stage that produced this artifact."""

    seed: int = 0
    prompt_hash: str = ""
    """SHA-256 of the positive prompt string (for reproducibility)."""

    # Quality
    qa_scores: Dict[str, float] = Field(default_factory=dict)
    """{'clip': 0.31, 'sharpness': 142.0, 'face_quality': 0.82}"""

    qa_passed: bool = True

    # Lifecycle
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    version: int = 1

    # Dependencies (for DAG-based incremental rebuild)
    depends_on: List[str] = Field(default_factory=list)
    """List of artifact_ids this artifact was generated from."""


class ArtifactStore:
    """
    Content-addressable artifact store for the pipeline.

    Thread-safety: Not thread-safe. Designed for single-process sequential use.
    """

    def __init__(self, workspace_dir: Path):
        self.workspace_dir = Path(workspace_dir)
        self._index_path = self.workspace_dir / "artifact_store.json"
        self._by_hash: Dict[str, ArtifactEntry] = {}
        self._by_shot: Dict[str, List[str]] = {}  # shot_id → [content_hash, ...]
        self._load()

    # ─────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────

    def register(
        self,
        path: str | Path,
        artifact_type: str,
        shot_id: str = "",
        scene_id: str = "",
        generator: str = "",
        seed: int = 0,
        prompt_hash: str = "",
        qa_scores: Optional[Dict[str, float]] = None,
        qa_passed: bool = True,
        depends_on: Optional[List[str]] = None,
    ) -> ArtifactEntry:
        """
        Register a newly generated artifact.

        If the file's content hash already exists in the store, the existing
        entry is returned (content-addressable deduplication).
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Cannot register non-existent file: {path}")

        content_hash = _file_sha256(p)
        artifact_id = content_hash[:16]

        if content_hash in self._by_hash:
            logger.debug(f"[ArtifactStore] Content-hash hit: {content_hash[:8]}… (skipping re-register)")
            return self._by_hash[content_hash]

        entry = ArtifactEntry(
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            path=str(p.absolute()),
            content_hash=content_hash,
            shot_id=shot_id,
            scene_id=scene_id,
            generator=generator,
            seed=seed,
            prompt_hash=prompt_hash,
            qa_scores=qa_scores or {},
            qa_passed=qa_passed,
            depends_on=depends_on or [],
        )

        self._by_hash[content_hash] = entry
        if shot_id:
            self._by_shot.setdefault(shot_id, []).append(content_hash)

        self._save()
        logger.debug(f"[ArtifactStore] Registered {artifact_type} for shot {shot_id!r}: {p.name}")
        return entry

    def get_best_for_shot(self, shot_id: str, artifact_type: str = "image") -> Optional[ArtifactEntry]:
        """
        Return the best QA-passing artifact for a given shot_id.

        'Best' = highest CLIP score among passing artifacts. Returns None if not found.
        """
        hashes = self._by_shot.get(shot_id, [])
        candidates = [
            self._by_hash[h]
            for h in hashes
            if h in self._by_hash
            and self._by_hash[h].artifact_type == artifact_type
            and self._by_hash[h].qa_passed
        ]
        if not candidates:
            return None
        # Sort by CLIP score desc, then recency
        candidates.sort(key=lambda e: (e.qa_scores.get("clip", 0.0), e.created_at), reverse=True)
        return candidates[0]

    def exists_for_shot(self, shot_id: str, artifact_type: str = "image") -> bool:
        """Return True if a QA-passing artifact already exists for this shot."""
        return self.get_best_for_shot(shot_id, artifact_type) is not None

    def find_by_prompt_hash(self, prompt_hash: str, artifact_type: str = "image") -> Optional[ArtifactEntry]:
        """
        Image Bank lookup: find an existing image generated from an identical prompt.

        This enables reuse across shots that happen to share the same prompt —
        saving GPU time on Kaggle.
        """
        for entry in self._by_hash.values():
            if (
                entry.prompt_hash == prompt_hash
                and entry.artifact_type == artifact_type
                and entry.qa_passed
            ):
                return entry
        return None

    def all_images_ordered(self) -> List[ArtifactEntry]:
        """Return all image artifacts ordered by shot_id for assembly."""
        images = [e for e in self._by_hash.values() if e.artifact_type == "image" and e.qa_passed]
        images.sort(key=lambda e: (e.shot_id, e.created_at))
        return images

    def summary(self) -> Dict[str, Any]:
        by_type: Dict[str, int] = {}
        for e in self._by_hash.values():
            by_type[e.artifact_type] = by_type.get(e.artifact_type, 0) + 1
        return {
            "total_artifacts": len(self._by_hash),
            "by_type": by_type,
            "shots_with_image": len([k for k in self._by_shot if self.exists_for_shot(k)]),
        }

    # ─────────────────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────────────────

    def _load(self):
        if self._index_path.exists():
            try:
                raw = json.loads(self._index_path.read_text(encoding="utf-8"))
                for h, data in raw.get("artifacts", {}).items():
                    self._by_hash[h] = ArtifactEntry(**data)
                self._by_shot = raw.get("by_shot", {})
                logger.info(f"[ArtifactStore] Loaded {len(self._by_hash)} artifacts from index.")
            except Exception as e:
                logger.warning(f"[ArtifactStore] Could not load index: {e}. Starting fresh.")

    def _save(self):
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "artifacts": {h: e.model_dump() for h, e in self._by_hash.items()},
            "by_shot": self._by_shot,
            "saved_at": datetime.utcnow().isoformat(),
        }
        self._index_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
