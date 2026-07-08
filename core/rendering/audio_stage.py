from core.pipeline.stage import PipelineStage, StageResult
from core.domain.assets.execution import ExecutionNode
from core.domain.scene.manifest import SceneManifest, ShotManifest
from core.domain.base import DomainModel
from pydantic import BaseModel
from typing import Dict
import os
import math
import hashlib

class AudioAsset(BaseModel):
    asset_id: str
    path: str
    duration: float
    text: str

class AudioManifest(DomainModel):
    voiceovers: Dict[str, AudioAsset] = {}  # Keyed by shot_id, so ffmpeg_renderer can
                                             # match each rendered image to its narration
                                             # by filename stem (<shot_id>.png / <shot_id>.wav).

class AudioGenerationStage(PipelineStage):
    def __init__(self, tts_provider=None, output_dir="workspace/audio"):
        self.tts = tts_provider
        self.output_dir = output_dir

    def get_providers(self) -> list:
        return [self.tts] if self.tts else []

    def execute(self, context) -> StageResult:
        if not self.tts:
            from plugins.audio.tts_provider import KokoroTTSProvider
            self.tts = KokoroTTSProvider()

        os.makedirs(self.output_dir, exist_ok=True)

        scene_manifest = None
        shot_manifest = None
        # Search in reverse to get the most recently produced version of each
        # manifest type (ShotManifest in particular is re-wrapped by CastPlanner
        # and CameraPlanner as the pipeline progresses).
        for node in reversed(context.execution_nodes):
            if scene_manifest is None and isinstance(node.artifact, SceneManifest):
                scene_manifest = node.artifact
            if shot_manifest is None and isinstance(node.artifact, ShotManifest):
                shot_manifest = node.artifact
            if scene_manifest is not None and shot_manifest is not None:
                break

        if not scene_manifest:
            raise ValueError("AudioGeneration: Missing SceneManifest.")
        if not shot_manifest:
            raise ValueError("AudioGeneration: Missing ShotManifest.")

        # Group shots by (scene_hash, beat_id). beat_id alone is not safe to use
        # as a global key: the LLM assigns beat ids independently per scene, so
        # "beat_1" in Scene A and "beat_1" in Scene B are unrelated and would
        # otherwise collide.
        shots_by_scene_beat: Dict[tuple, list] = {}
        for shot in shot_manifest.shots:
            parts = shot.shot_id.split("_")
            scene_hash = parts[1] if len(parts) > 1 else ""
            shots_by_scene_beat.setdefault((scene_hash, shot.beat_id), []).append(shot)

        manifest = AudioManifest()

        for scene in scene_manifest.scenes:
            scene_hash = hashlib.sha256(scene.scene_id.encode('utf-8')).hexdigest()[:8]

            for beat in scene.beats:
                shots_for_beat = shots_by_scene_beat.get((scene_hash, beat.beat_id), [])

                if not shots_for_beat:
                    # ShotPlanner didn't echo back a matching beat_id for this beat
                    # (LLM inconsistency). Still synthesize the narration and save
                    # it under beat_id so nothing is silently lost, even though
                    # ffmpeg_renderer won't be able to attach it to a specific shot.
                    self._synthesize(
                        manifest, key=beat.beat_id, text=beat.description or "...",
                        asset_id=f"audio_{beat.beat_id}",
                    )
                    continue

                # Split the beat's narration text proportionally across its shots
                # so the full line is spoken once (in order) as the video cuts
                # between that beat's several camera angles, rather than being
                # repeated in full for every shot.
                words = (beat.description or "").split()
                n = len(shots_for_beat)
                chunk_size = max(1, math.ceil(len(words) / n)) if words else 0

                for idx, shot in enumerate(shots_for_beat):
                    start = idx * chunk_size
                    end = start + chunk_size
                    chunk_words = words[start:end] if words else []
                    text = " ".join(chunk_words) if chunk_words else (beat.description or "...")

                    self._synthesize(
                        manifest, key=shot.shot_id, text=text,
                        asset_id=f"audio_{shot.shot_id}",
                    )

        node = ExecutionNode(artifact=manifest, stage_name="AudioGenerationStage")

        return StageResult(
            artifact=manifest,
            execution_node=node,
            metrics={"audio_generated": len(manifest.voiceovers)},
            metadata={}
        )

    def _synthesize(self, manifest: AudioManifest, key: str, text: str, asset_id: str):
        filename = f"{key}.wav"
        output_path = os.path.join(self.output_dir, filename)

        duration = self.tts.generate_voice(text=text, voice_id="default", output_path=output_path)

        manifest.voiceovers[key] = AudioAsset(
            asset_id=asset_id,
            path=output_path,
            duration=duration,
            text=text,
        )
