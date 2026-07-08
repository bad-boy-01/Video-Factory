from core.pipeline.stage import PipelineStage, StageResult
from core.domain.assets.execution import ExecutionNode
from core.domain.prompt.ast import PromptManifest
from core.domain.scene.manifest import SceneManifest
from core.rendering.audio_stage import AudioManifest
from core.domain.timeline.models import Timeline, TimelineTrack, TimelineClip, Animation
import logging
import hashlib
import os

logger = logging.getLogger(__name__)

class TimelineBuilderStage(PipelineStage):
    def __init__(self, output_dir="workspace"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def get_providers(self) -> list:
        return []

    def execute(self, context) -> StageResult:
        prompt_manifest = None
        scene_manifest = None
        audio_manifest = None
        
        for node in context.execution_nodes:
            if isinstance(node.artifact, PromptManifest):
                prompt_manifest = node.artifact
            elif isinstance(node.artifact, SceneManifest):
                scene_manifest = node.artifact
            elif isinstance(node.artifact, AudioManifest):
                audio_manifest = node.artifact
                
        if not prompt_manifest or not scene_manifest:
            raise ValueError("TimelineBuilder: Missing PromptManifest or SceneManifest.")
            
        video_track = TimelineTrack(track_id="video_main", type="video", z_index=0)
        subtitle_track = TimelineTrack(track_id="subtitles", type="subtitle", z_index=10)
        voice_track = TimelineTrack(track_id="voice", type="voice")
        
        current_time = 0.0
        
        for p in prompt_manifest.prompts:
            # Look up audio duration if AudioManifest exists
            duration = 4.0 # default fallback
            beat_id = p.shot_id.replace("shot_", "beat_") # naive mapping for scaffolding
            
            audio_text = "Subtitle fallback"
            
            if audio_manifest and beat_id in audio_manifest.voiceovers:
                asset = audio_manifest.voiceovers[beat_id]
                duration = asset.duration
                audio_text = asset.text
                
                # Add voice clip
                voice_clip = TimelineClip(
                    clip_id=f"vo_{p.prompt_id}",
                    asset_id=asset.asset_id,
                    start_time=current_time,
                    end_time=current_time + duration
                )
                voice_track.clips.append(voice_clip)
                
            # Add video clip
            anim = Animation(type=p.ast.camera.movement, duration=duration)
            
            video_clip = TimelineClip(
                clip_id=f"vid_{p.prompt_id}",
                asset_id=f"asset_{p.shot_id}",
                start_time=current_time,
                end_time=current_time + duration,
                animation=anim
            )
            video_track.clips.append(video_clip)
            
            # Add subtitle clip
            sub_clip = TimelineClip(
                clip_id=f"sub_{p.prompt_id}",
                asset_id="",
                start_time=current_time,
                end_time=current_time + duration,
                text=audio_text
            )
            subtitle_track.clips.append(sub_clip)
            
            current_time += duration
            
        # Deterministic checksum of timeline contents
        checksum_input = f"{len(video_track.clips)}_{current_time}"
        checksum = hashlib.md5(checksum_input.encode('utf-8')).hexdigest()
        
        timeline = Timeline(
            checksum=checksum,
            generated_from_prompt_manifest_hash=getattr(prompt_manifest, "source_hash", "") or "",
            generated_from_scene_manifest_hash=getattr(scene_manifest, "source_hash", "") or "",
            tracks={
                "video_main": video_track,
                "subtitles": subtitle_track,
                "voice": voice_track
            },
            generator="TimelineBuilderStage"
        )
        
        node = ExecutionNode(artifact=timeline, stage_name="TimelineBuilderStage")
        
        return StageResult(
            artifact=timeline,
            execution_node=node,
            metrics={"duration": current_time, "video_clips": len(video_track.clips)},
            metadata={}
        )
