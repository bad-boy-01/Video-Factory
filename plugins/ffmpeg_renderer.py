"""
FFmpegVideoRenderer — upgraded with Ken Burns, xfade transitions, and subtitle burn-in.

Pipeline (in order):
  1. For each still image → generate a Ken Burns motion clip (zoompan, 25fps)
     The zoom direction is determined by shot.movement (push_in, pull_back, pan_left, pan_right, static)
  2. Concatenate all motion clips with 0.5s xfade crossfade transitions
  3. Normalize audio (2-pass loudnorm) — kept from original
  4. Mix narration audio with video
  5. Burn in subtitles from SRT file (if available)
  6. Final output: final_video.mp4

Ken Burns zoompan filter reference:
  push_in:    slow zoom toward center
  pull_back:  slow zoom away from center
  pan_left:   slow horizontal drift left
  pan_right:  slow horizontal drift right
  static:     absolutely no motion (for extreme tension beats)

Subtitle font: system Noto Sans (good Unicode coverage for manhwa feel).
Falls back to default font if Noto Sans not found.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Dict, List, Optional

from core.domain.assets.execution import FrameManifest
from plugins.interfaces import VideoRendererProvider

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

TARGET_FPS = 25
DEFAULT_RESOLUTION = "1024x1024"
DEFAULT_DURATION = 3.0
MIN_DURATION = 0.5

# Zoompan expressions per movement type.
# {fps} and {dur} are template placeholders replaced before calling FFmpeg.
ZOOMPAN_EXPRS: Dict[str, str] = {
    "push_in":   "z='min(zoom+0.0015,1.12)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'",
    "pull_back": "z='if(eq(on\\,1)\\,1.12\\,max(zoom-0.0015\\,1.0))':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'",
    "pan_left":  "z='1.05':x='iw/zoom/2+((iw-iw/zoom)/2)*(on/({total_frames}))':y='ih/2-(ih/zoom/2)'",
    "pan_right": "z='1.05':x='iw/zoom/2-((iw-iw/zoom)/2)*(on/({total_frames}))':y='ih/2-(ih/zoom/2)'",
    "crane_up":  "z='1.05':x='iw/2-(iw/zoom/2)':y='ih/zoom/2+((ih-ih/zoom)/2)*(on/({total_frames}))'",
    "dolly_in":  "z='min(zoom+0.001,1.08)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'",
    "whip_pan":  "z='1.0':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'",  # Static — whip pan needs true video
    "static":    "z='1.0':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'",
}


class FFmpegVideoRenderer(VideoRendererProvider):
    """
    Full cinematic FFmpeg renderer with Ken Burns, xfade transitions, and subtitle burn-in.
    """

    def __init__(self, resolution: str = DEFAULT_RESOLUTION, fps: int = TARGET_FPS):
        self.resolution = resolution
        self.fps = fps

    def render_video(
        self,
        manifest: FrameManifest,
        audio_paths: List[Path],
        output_path: Path,
        subtitle_path: Optional[Path] = None,
        shot_movements: Optional[Dict[str, str]] = None,
    ) -> Path:
        """
        Full pipeline: still images → Ken Burns clips → xfade concat →
        audio mix → subtitle burn-in → final video.

        Parameters
        ----------
        manifest:
            Ordered list of frame entries (shot_id + image_path + duration).
        audio_paths:
            List of per-shot WAV files for narration.
        output_path:
            Destination path for final_video.mp4.
        subtitle_path:
            Optional .srt file for subtitle burn-in.
        shot_movements:
            Optional dict of shot_id → movement type for Ken Burns direction.
            Defaults to 'push_in' for all shots if not provided.
        """
        if not manifest.frames:
            raise ValueError("FrameManifest is empty. Cannot render video.")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        shot_movements = shot_movements or {}
        audio_map = {p.stem: p for p in audio_paths}

        # ── Step 1: Generate Ken Burns clip for each still image ──────────
        with tempfile.TemporaryDirectory(prefix="novelfactory_kb_") as tmpdir:
            tmp = Path(tmpdir)
            clip_paths: List[Path] = []
            clip_durations: List[float] = []

            for entry in manifest.frames:
                image_path = Path(entry.image_path)
                if not image_path.exists():
                    logger.warning(f"[FFmpeg] Image not found: {image_path}. Skipping.")
                    continue

                # Determine duration
                duration = entry.duration if entry.duration > MIN_DURATION else DEFAULT_DURATION
                if entry.shot_id in audio_map:
                    try:
                        with wave.open(str(audio_map[entry.shot_id]), "r") as w:
                            duration = max(MIN_DURATION, w.getnframes() / w.getframerate())
                    except Exception:
                        pass

                movement = shot_movements.get(entry.shot_id, "push_in")
                clip_path = tmp / f"{entry.shot_id}_kb.mp4"

                success = self._generate_ken_burns_clip(
                    image_path, clip_path, movement, duration
                )
                if success:
                    clip_paths.append(clip_path)
                    clip_durations.append(duration)
                else:
                    logger.warning(f"[FFmpeg] Ken Burns failed for {entry.shot_id}. Using raw concat fallback.")
                    # Fallback: static clip
                    static_path = tmp / f"{entry.shot_id}_static.mp4"
                    self._generate_static_clip(image_path, static_path, duration)
                    clip_paths.append(static_path)
                    clip_durations.append(duration)

            if not clip_paths:
                raise ValueError("[FFmpeg] No valid clips generated.")

            # ── Step 2: Concatenate clips with xfade transitions ──────────
            concat_output = tmp / "concat_video.mp4"
            self._concat_with_xfade(clip_paths, clip_durations, concat_output)

            # ── Step 3: Mix audio ─────────────────────────────────────────
            audio_output = output_path.parent / "pre_subtitle_video.mp4"
            if audio_paths:
                self._mix_audio(concat_output, manifest, audio_map, audio_output)
            else:
                import shutil
                shutil.copy(str(concat_output), str(audio_output))

            # ── Step 4: Burn in subtitles ─────────────────────────────────
            if subtitle_path and subtitle_path.exists():
                self._burn_subtitles(audio_output, subtitle_path, output_path)
                audio_output.unlink(missing_ok=True)
            else:
                import shutil
                shutil.move(str(audio_output), str(output_path))

        logger.info(f"[FFmpeg] Final render complete: {output_path}")
        return output_path

    # ─────────────────────────────────────────────────────────────────────
    # Ken Burns
    # ─────────────────────────────────────────────────────────────────────

    def _generate_ken_burns_clip(
        self, image_path: Path, output_path: Path, movement: str, duration: float
    ) -> bool:
        """Generate a zoompan motion clip from a still image."""
        total_frames = max(1, int(duration * self.fps))
        expr_template = ZOOMPAN_EXPRS.get(movement, ZOOMPAN_EXPRS["push_in"])
        zoompan_expr = expr_template.format(total_frames=total_frames)

        cmd = [
            "ffmpeg", "-y",
            "-loop", "1",
            "-i", str(image_path),
            "-vf", (
                f"zoompan={zoompan_expr}:d={total_frames}:s={self.resolution},"
                f"fps={self.fps}"
            ),
            "-t", str(duration),
            "-pix_fmt", "yuv420p",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            str(output_path),
        ]
        return self._run(cmd, f"Ken Burns [{movement}] {image_path.name}")

    def _generate_static_clip(self, image_path: Path, output_path: Path, duration: float) -> bool:
        """Fallback: static clip with no motion."""
        cmd = [
            "ffmpeg", "-y",
            "-loop", "1",
            "-i", str(image_path),
            "-vf", f"scale={self.resolution},fps={self.fps}",
            "-t", str(duration),
            "-pix_fmt", "yuv420p",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            str(output_path),
        ]
        return self._run(cmd, f"Static clip {image_path.name}")

    # ─────────────────────────────────────────────────────────────────────
    # xfade transitions
    # ─────────────────────────────────────────────────────────────────────

    def _concat_with_xfade(
        self,
        clip_paths: List[Path],
        clip_durations: List[float],
        output_path: Path,
        fade_duration: float = 0.5,
    ) -> bool:
        """Chain xfade filters between clips. Falls back to simple concat if only 1 clip."""
        if len(clip_paths) == 1:
            import shutil
            shutil.copy(str(clip_paths[0]), str(output_path))
            return True

        if len(clip_paths) == 2:
            # Special case: just one xfade
            offset = max(0.01, clip_durations[0] - fade_duration)
            inputs = ["-i", str(clip_paths[0]), "-i", str(clip_paths[1])]
            cmd = (
                ["ffmpeg", "-y"] + inputs +
                ["-filter_complex",
                 f"[0:v][1:v]xfade=transition=fade:duration={fade_duration}:offset={offset}[v]",
                 "-map", "[v]", "-pix_fmt", "yuv420p", "-c:v", "libx264",
                 "-preset", "fast", "-crf", "18", str(output_path)]
            )
            return self._run(cmd, "xfade 2-clip")

        # General case: build chained xfade filter
        inputs_args = []
        for p in clip_paths:
            inputs_args += ["-i", str(p)]

        filter_parts = []
        running_offset = 0.0
        current_label = "[0:v]"

        for i in range(1, len(clip_paths)):
            running_offset += clip_durations[i - 1] - fade_duration
            running_offset = max(0.01, running_offset)
            out_label = f"[v{i}]" if i < len(clip_paths) - 1 else "[vout]"
            filter_parts.append(
                f"{current_label}[{i}:v]xfade=transition=fade"
                f":duration={fade_duration}:offset={running_offset:.3f}{out_label}"
            )
            current_label = out_label

        filter_complex = ";".join(filter_parts)
        cmd = (
            ["ffmpeg", "-y"] + inputs_args +
            ["-filter_complex", filter_complex,
             "-map", "[vout]",
             "-pix_fmt", "yuv420p", "-c:v", "libx264",
             "-preset", "fast", "-crf", "18", str(output_path)]
        )
        return self._run(cmd, f"xfade {len(clip_paths)}-clip chain")

    # ─────────────────────────────────────────────────────────────────────
    # Audio mixing (preserved from original with improvements)
    # ─────────────────────────────────────────────────────────────────────

    def _mix_audio(
        self,
        video_path: Path,
        manifest: FrameManifest,
        audio_map: Dict[str, Path],
        output_path: Path,
    ) -> bool:
        """2-pass loudnorm audio mix. Identical logic to the original FFmpegVideoRenderer."""
        audio_concat_file = output_path.parent / "audio_concat.txt"
        with open(audio_concat_file, "w") as f:
            for entry in manifest.frames:
                if entry.shot_id in audio_map:
                    f.write(f"file '{audio_map[entry.shot_id].absolute().as_posix()}'\n")

        # Pass 1: measure loudness
        measure_cmd = [
            "ffmpeg", "-hide_banner",
            "-f", "concat", "-safe", "0", "-i", str(audio_concat_file),
            "-af", "loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json",
            "-f", "null", "-",
        ]
        measured_i, measured_tp, measured_lra, measured_thresh, measured_offset = \
            "-24", "-2", "7", "-34", "0"

        try:
            result = subprocess.run(
                measure_cmd, check=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            stderr = result.stderr
            js = stderr[stderr.find("{"):stderr.rfind("}") + 1]
            if js:
                d = json.loads(js)
                measured_i = d.get("input_i", measured_i)
                measured_tp = d.get("input_tp", measured_tp)
                measured_lra = d.get("input_lra", measured_lra)
                measured_thresh = d.get("input_thresh", measured_thresh)
                measured_offset = d.get("target_offset", measured_offset)
        except Exception as e:
            logger.warning(f"[FFmpeg] Loudness measurement failed: {e}. Using defaults.")

        # Pass 2: normalize + mix
        mix_cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-f", "concat", "-safe", "0", "-i", str(audio_concat_file),
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "copy",
            "-af", (
                f"loudnorm=I=-16:TP=-1.5:LRA=11"
                f":measured_I={measured_i}:measured_TP={measured_tp}"
                f":measured_LRA={measured_lra}:measured_thresh={measured_thresh}"
                f":offset={measured_offset}:linear=true:print_format=summary,"
                "aformat=sample_rates=48000:channel_layouts=stereo"
            ),
            "-c:a", "aac",
            str(output_path),
        ]
        return self._run(mix_cmd, "audio mix")

    # ─────────────────────────────────────────────────────────────────────
    # Subtitle burn-in
    # ─────────────────────────────────────────────────────────────────────

    def _burn_subtitles(
        self, video_path: Path, subtitle_path: Path, output_path: Path
    ) -> bool:
        """Burn SRT subtitles into the video using FFmpeg drawtext/subtitles filter."""
        # Escape path for FFmpeg on Windows (forward slashes, escaped colons)
        srt_escaped = str(subtitle_path.absolute()).replace("\\", "/").replace(":", "\\:")

        style = (
            "FontName=Arial,"
            "FontSize=22,"
            "PrimaryColour=&H00FFFFFF,"   # White text
            "OutlineColour=&H00000000,"   # Black outline
            "BackColour=&H80000000,"      # Semi-transparent background box
            "Outline=1,"
            "Shadow=1,"
            "Bold=1,"
            "Alignment=2"                 # Bottom center
        )

        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vf", f"subtitles='{srt_escaped}':force_style='{style}'",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            "-c:a", "copy",
            str(output_path),
        ]
        return self._run(cmd, "subtitle burn-in")

    # ─────────────────────────────────────────────────────────────────────
    # Shared subprocess runner
    # ─────────────────────────────────────────────────────────────────────

    def _run(self, cmd: List[str], label: str) -> bool:
        try:
            logger.info(f"[FFmpeg] {label}...")
            subprocess.run(
                cmd, check=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"[FFmpeg] {label} FAILED:\n{e.stderr.decode(errors='replace')}")
            return False
