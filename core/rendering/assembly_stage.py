from core.pipeline.stage import PipelineStage, StageResult
from core.domain.assets.execution import ExecutionNode
from core.domain.timeline.models import Timeline
from core.rendering.render_queue import RenderQueue
import os
import json
import logging
import subprocess

logger = logging.getLogger(__name__)

class FFmpegAssemblyStage(PipelineStage):
    def __init__(self, output_dir="workspace"):
        self.output_dir = output_dir

    def get_providers(self) -> list:
        return []
        
    def execute(self, context) -> StageResult:
        timeline = None
        for node in context.execution_nodes:
            if isinstance(node.artifact, Timeline):
                timeline = node.artifact
                break
                
        if not timeline:
            raise ValueError("FFmpegAssembly: Missing Timeline.")
            
        registry = context.registry
        workspace = context.workspace
        
        # Get count of video clips
        video_track = timeline.tracks.get("video_main")
        clip_count = len(video_track.clips) if video_track else 0
        logger.info(f"Assembling video from Timeline ({clip_count} items)...")
        
        # 1. Create concat file for FFmpeg
        concat_file = workspace.get_output_path("concat.txt")
        
        images = []
        if video_track:
            for clip in video_track.clips:
                asset = registry.assets.get(clip.asset_id)
                if asset and os.path.exists(asset.path):
                    images.append(asset.path)
        
        if not images:
            raise ValueError("FFmpegAssembly: No rendered assets found.")
            
        with open(concat_file, "w", encoding="utf-8") as f:
            for img in images:
                # Use absolute paths and forward slashes for FFmpeg path compatibility
                img_path = os.path.abspath(img).replace("\\", "/")
                f.write(f"file '{img_path}'\n")
                f.write(f"duration 4.0\n") # Static duration for demo
                
        output_video = workspace.get_output_path("final_video.mp4")
        srt_file = workspace.get_output_path("subtitles.srt")
        
        # 2. FFmpeg command: Subtitle multiplexing
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_file)
        ]
        
        if os.path.exists(srt_file):
            # Escape path for FFmpeg filter
            srt_escaped = srt_file.replace("\\", "/").replace(":", "\\:")
            cmd.extend(["-vf", f"subtitles={srt_escaped}"])
            
        cmd.extend([
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            output_video
        ])
        
        logger.info(f"Running FFmpeg: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"Successfully rendered video: {output_video}")
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg failed: {e.stderr.decode()}")
            raise e
            
        # 3. Cleanup: Images are archived in AssetRegistry, do not delete them.
        logger.info("Preserving intermediate assets in cache (Archive-based management).")
            
        RenderQueue().vacuum()
        logger.info("RenderQueue vacuumed.")
        
        node = ExecutionNode(artifact=timeline, stage_name="FFmpegAssemblyStage")
        
        return StageResult(
            artifact=timeline,
            execution_node=node,
            metrics={"video_path": output_video},
            metadata={}
        )
