"""
Character History Store
========================
Keeps every version of a character/location profile seen during StoryBible
extraction, not just the final merged one.

Why this exists: StoryBibleGeneratorStage's dedup logic keeps whichever
extraction of a character has more populated fields ("richer wins"). That
heuristic has a real blind spot - a later chunk's extraction can be richer
*and wrong* (an LLM hallucination introduced while re-describing a character
30 chapters later), silently overwriting a correct earlier description with
no way to notice or undo it. This store keeps the full history so a person
can inspect what changed and roll back to an earlier version if a later one
turns out to be worse, without re-running the whole planning stage.

Persisted at: workspace/manifests/character_history.json
"""

import json
import logging
import os
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class CharacterHistoryStore:
    def __init__(self, history_path: str):
        self.history_path = history_path
        self._data = {"characters": {}, "locations": {}}
        self._load()

    # ── Recording new versions (called during StoryBible extraction) ──────────

    def record(self, kind: str, entity_id: str, chunk_id: str, profile: dict, is_active: bool):
        """
        kind: "characters" or "locations"
        Appends a new version and, if is_active, marks it as the current one.
        """
        bucket = self._data.setdefault(kind, {})
        entry = bucket.setdefault(entity_id, {"active_version": 0, "versions": []})
        entry["versions"].append({"chunk_id": chunk_id, "profile": profile})
        if is_active:
            entry["active_version"] = len(entry["versions"]) - 1

    def save(self):
        try:
            os.makedirs(os.path.dirname(self.history_path), exist_ok=True)
            with open(self.history_path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"[CharHistory] Failed to save history: {e}")

    # ── Inspection ──────────────────────────────────────────────────────────

    def list_versions(self, kind: str, entity_id: str) -> Optional[dict]:
        """Returns {"active_version": int, "versions": [...]}, or None if unknown."""
        return self._data.get(kind, {}).get(entity_id)

    def all_entities(self, kind: str) -> List[str]:
        return list(self._data.get(kind, {}).keys())

    # ── Rollback ────────────────────────────────────────────────────────────

    def rollback(self, kind: str, entity_id: str, version_index: int) -> dict:
        """
        Sets version_index as the active version for entity_id and persists it.
        Returns the profile dict at that version.
        Raises KeyError/IndexError with a clear message if entity_id or
        version_index don't exist.
        """
        bucket = self._data.get(kind, {})
        if entity_id not in bucket:
            raise KeyError(f"'{entity_id}' has no recorded history under '{kind}'.")

        entry = bucket[entity_id]
        versions = entry["versions"]
        if not (0 <= version_index < len(versions)):
            raise IndexError(
                f"'{entity_id}' has {len(versions)} version(s) (valid indices "
                f"0..{len(versions) - 1}); {version_index} is out of range."
            )

        entry["active_version"] = version_index
        self.save()
        return versions[version_index]["profile"]

    # ── Persistence ─────────────────────────────────────────────────────────

    def _load(self):
        if os.path.exists(self.history_path):
            try:
                with open(self.history_path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
            except Exception as e:
                logger.warning(f"[CharHistory] Failed to load existing history: {e}")
