"""
Shared utility: deterministic scene hash.

Centralises the hashlib.sha256 computation that was previously duplicated
verbatim in shot_planner.py, cast_planner.py, audio_stage.py, and
prompt_builder.py.
"""
import hashlib


def compute_scene_hash(scene_id: str, length: int = 8) -> str:
    """Return a short hex digest for `scene_id` suitable for use in shot IDs."""
    return hashlib.sha256(scene_id.encode("utf-8")).hexdigest()[:length]


def compute_content_hash(content: str) -> str:
    """Return a full SHA-256 hex digest for arbitrary content (used in ArtifactStore)."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def compute_file_hash(path) -> str:
    """Return a SHA-256 hex digest for a file's raw bytes."""
    import os
    from pathlib import Path
    p = Path(path)
    if not p.exists():
        return ""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
