import subprocess
from pathlib import Path
from core.domain.assets.execution import FrameManifest
from plugins.interfaces import VideoRendererProvider
import wave
import os

class FFmpegVideoRenderer(VideoRendererProvider):
    def render_video(self, manifest: FrameManifest, audio_paths: list[Path], output_path: Path) -> Path:
        """
        Consumes a strictly ordered FrameManifest, normalizes the images, 
        and invokes FFmpeg to stitch them together.
        """
        if not manifest.frames:
            raise ValueError("FrameManifest is empty. Cannot render video.")
            
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        audio_map = {p.stem: p for p in audio_paths}
            
        # 1. Create a concat list file
        concat_file = output_path.parent / "concat.txt"
        with open(concat_file, "w") as f:
            for entry in manifest.frames:
                # Normalizing paths for ffmpeg (forward slashes even on Windows)
                safe_path = entry.image_path.absolute().as_posix()
                
                duration = 3.0
                if entry.shot_id in audio_map:
                    wav_path = audio_map[entry.shot_id]
                    try:
                        with wave.open(str(wav_path), 'r') as w:
                            duration = max(2.0, w.getnframes() / w.getframerate())
                    except Exception:
                        duration = 3.0
                        
                f.write(f"file '{safe_path}'\n")
                f.write(f"duration {duration}\n")
        
        silent_output = output_path.parent / "silent_video.mp4"
        
        # 2. Invoke FFmpeg to create silent video.
        # Deliberately NOT using -vsync vfr here: at the very low effective frame
        # rates this concat list produces (one "frame" per multi-second shot),
        # -vsync vfr has been observed to drop the entire first entry (every
        # subsequent shot shifts one slot earlier, and the last shot silently
        # absorbs all the leftover duration). Omitting it falls back to standard
        # CFR output (ffmpeg duplicates each image across enough real frames to
        # fill its requested duration), which handles the concat list correctly,
        # including the final entry's duration - so no extra workaround is
        # needed for that either.
        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", 
            "-i", str(concat_file),
            "-pix_fmt", "yuv420p",
            str(silent_output)
        ]
        
        try:
            print("[FFMPEG] Rendering silent video...")
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(f"[FFMPEG] Silent render complete: {silent_output}")
        except subprocess.CalledProcessError as e:
            print("[FFMPEG] Render failed.")
            print(e.stderr.decode())
            raise e
            
        # 3. Mix audio if available
        if audio_paths:
            audio_concat_file = output_path.parent / "audio_concat.txt"
            with open(audio_concat_file, "w") as f:
                for entry in manifest.frames:
                    if entry.shot_id in audio_map:
                        f.write(f"file '{audio_map[entry.shot_id].absolute().as_posix()}'\n")

            # Explicit -map avoids relying on ffmpeg's automatic stream
            # selection across two inputs. -shortest is deliberately omitted:
            # combined with this concat-demuxer video + concat-demuxer raw-PCM
            # audio, it was observed to truncate the video stream to almost
            # nothing (0 bytes of video data written). Per-shot duration is
            # already derived from that shot's own audio, so video and audio
            # are already closely aligned without it.
            mix_cmd = [
                "ffmpeg", "-y", 
                "-i", str(silent_output),
                "-f", "concat", "-safe", "0", "-i", str(audio_concat_file),
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-c:v", "copy",
                "-c:a", "aac",
                str(output_path)
            ]
            try:
                print("[FFMPEG] Mixing audio...")
                subprocess.run(mix_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                print(f"[FFMPEG] Final render complete: {output_path}")
            except subprocess.CalledProcessError as e:
                print("[FFMPEG] Audio mix failed.")
                print(e.stderr.decode())
                raise e
        else:
            # If no audio, just rename silent to final
            os.replace(str(silent_output), str(output_path))
            
        return output_path
