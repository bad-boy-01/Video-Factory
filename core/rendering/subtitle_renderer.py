"""
SRT subtitle renderer — generates an SRT file from AudioManifest voiceovers.

The SRT file is then passed to FFmpegVideoRenderer for subtitle burn-in.

Timing is derived from:
  1. AudioManifest.voiceovers[shot_id].duration  (actual TTS audio length)
  2. Shot ordering from FrameManifest (preserved order)

If a shot has no voiceover, its subtitle entry is skipped.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def _format_srt_time(seconds: float) -> str:
    """Convert seconds to SRT timestamp format: HH:MM:SS,mmm"""
    total_ms = int(seconds * 1000)
    ms = total_ms % 1000
    total_s = total_ms // 1000
    secs = total_s % 60
    total_m = total_s // 60
    mins = total_m % 60
    hours = total_m // 60
    return f"{hours:02d}:{mins:02d}:{secs:02d},{ms:03d}"


def generate_srt(
    shot_ids: List[str],
    shot_durations: Dict[str, float],
    voiceover_texts: Dict[str, str],
    output_path: Path,
) -> Path:
    """
    Generate an SRT subtitle file.

    Parameters
    ----------
    shot_ids:
        Ordered list of shot IDs (defines subtitle display order).
    shot_durations:
        shot_id → display duration in seconds.
    voiceover_texts:
        shot_id → subtitle text. Shots without an entry are skipped.
    output_path:
        Where to write the .srt file.

    Returns
    -------
    Path to the written SRT file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    index = 1
    current_time = 0.0

    for shot_id in shot_ids:
        duration = shot_durations.get(shot_id, 3.0)
        text = voiceover_texts.get(shot_id, "")

        if text.strip():
            start_ts = _format_srt_time(current_time)
            end_ts = _format_srt_time(current_time + duration)
            lines.append(f"{index}")
            lines.append(f"{start_ts} --> {end_ts}")
            lines.append(text.strip())
            lines.append("")  # Blank line between entries
            index += 1

        current_time += duration

    srt_content = "\n".join(lines)
    output_path.write_text(srt_content, encoding="utf-8")
    logger.info(f"[SubtitleRenderer] Generated {index - 1} subtitle entries → {output_path}")
    return output_path


def generate_srt_from_audio_manifest(
    audio_manifest,
    shot_ids: List[str],
    output_path: Path,
) -> Optional[Path]:
    """
    Convenience wrapper: generate SRT from an AudioManifest object.

    Parameters
    ----------
    audio_manifest:
        AudioManifest with a .voiceovers dict (shot_id → VoiceoverAsset).
    shot_ids:
        Ordered shot IDs from FrameManifest.
    output_path:
        Where to write the .srt file.
    """
    if not audio_manifest or not hasattr(audio_manifest, "voiceovers"):
        logger.warning("[SubtitleRenderer] No AudioManifest or voiceovers found; skipping SRT.")
        return None

    shot_durations: Dict[str, float] = {}
    voiceover_texts: Dict[str, str] = {}

    for shot_id in shot_ids:
        asset = audio_manifest.voiceovers.get(shot_id)
        if asset:
            shot_durations[shot_id] = getattr(asset, "duration", 3.0)
            voiceover_texts[shot_id] = getattr(asset, "text", "")
        else:
            shot_durations[shot_id] = 3.0

    return generate_srt(shot_ids, shot_durations, voiceover_texts, output_path)
