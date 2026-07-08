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
        
        # 1. Create a directory for the manhwa panels
        slideshow_dir = workspace.get_output_path("manhwa_slideshow")
        slideshow_dir.mkdir(parents=True, exist_ok=True)
        
        images = []
        if video_track:
            for clip in video_track.clips:
                asset = registry.assets.get(clip.asset_id)
                if asset and os.path.exists(asset.path):
                    images.append(asset.path)
        
        if not images:
            raise ValueError("ManhwaAssembly: No rendered assets found.")
            
        import shutil
        panel_paths = []
        for i, img_path in enumerate(images):
            # Output nicely numbered panels
            dest_name = f"panel_{i:03d}.png"
            dest_path = slideshow_dir / dest_name
            shutil.copy(img_path, dest_path)
            panel_paths.append(str(dest_path))
            
        logger.info(f"Successfully assembled {len(images)} manhwa panels in {slideshow_dir}")
            
        # 2. Cleanup: Images are archived in AssetRegistry, do not delete them.
        logger.info("Preserving intermediate assets in cache (Archive-based management).")
            
        RenderQueue().vacuum()
        logger.info("RenderQueue vacuumed.")
        
        node = ExecutionNode(artifact=timeline, stage_name="ManhwaAssemblyStage")
        
        return StageResult(
            artifact=timeline,
            execution_node=node,
            metrics={"slideshow_dir": str(slideshow_dir), "panel_count": len(panel_paths)},
            metadata={}
        )
