from core.pipeline.stage import CompilerStage, StageResult
from core.domain.assets.execution import ExecutionNode
from core.domain.scene.manifest import SceneManifest, ShotManifest, Shot
from core.pipeline.context import PipelineContext
import hashlib
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

class ShotPlannerStage(CompilerStage):
    def __init__(self, llm_provider=None):
        self.llm = llm_provider

    def get_name(self) -> str:
        return "ShotPlannerStage"

    def get_providers(self) -> list:
        return [self.llm] if self.llm else []
        
    def inputs(self, context: PipelineContext) -> list[Any]:
        # ShotPlanner depends on SceneManifest
        for node in context.execution_nodes:
            if isinstance(node.artifact, SceneManifest):
                return [node.artifact]
        return []
        
    def outputs(self) -> list[str]:
        return ["shot_manifest"]
        
    def generator_signature(self) -> str:
        return f"{self.get_name()}_{type(self.llm).__name__ if self.llm else 'default'}_v1.0"

    def execute(self, context: PipelineContext) -> StageResult:
        if not self.llm:
            from plugins.local_llm import LocalLLMProvider
            self.llm = LocalLLMProvider()
            
        scene_manifest = None
        for node in context.execution_nodes:
            if isinstance(node.artifact, SceneManifest):
                scene_manifest = node.artifact
                break
                
        if not scene_manifest:
            raise ValueError("No SceneManifest found in context.")
            
        schema = {
            "scene_style": "string (narrative | motion | montage)",
            "beats": [
                {
                    "beat_id": "string",
                    "shots": [
                        {
                            "purpose": "string (e.g. establishing, reaction, insert, closeup)",
                            "emotion": "string",
                            "importance": "string (high, medium, low)",
                            "focus": "string (environment, character, object)",
                            "duration": 3.0
                        }
                    ]
                }
            ]
        }
            
        all_shots = []
        
        for scene_idx, scene in enumerate(scene_manifest.scenes):
            scene_hash = hashlib.sha256(scene.scene_id.encode('utf-8')).hexdigest()[:8]
            
            prompt = f"""
You are a master cinematographer and film editor. Expand the following narrative beats into a cinematic shot sequence (coverage).
CRITICAL RULES:
- First, choose one of three Cinematic Styles for this scene based on the beats:
  1. NARRATIVE: Focuses on dialogue, character interaction, and story progression. Use standard coverage (Establishing -> Medium -> Close-ups/Reactions). Duration: 2.0s - 4.0s.
  2. MOTION/ACTION: Focuses on speed, kinetic clarity, and physical conflict. Use rapid tracking, intense pacing, and fast cuts. Duration: 1.0s - 2.5s.
  3. MONTAGE: Focuses on internal emotion and passage of time via visual juxtaposition. Limit dialogue, focus on objects/expressions. Duration: 1.0s - 2.0s.
- Provide 3 to 6 shots per beat depending on the chosen style.
- Start a scene with an Establishing shot.
- Use distinct purposes like 'Reaction', 'Over shoulder', 'Insert object', 'Close-up'.
- Do NOT output the physical camera parameters (like 50mm or eye-level). Output semantic intent: purpose, emotion, importance, and focus.

Scene Location: {scene.location}
Scene Emotion: {scene.emotion}
Characters Present: {', '.join(scene.characters)}

Beats:
"""
            for beat in scene.beats:
                prompt += f"- [Beat {beat.beat_id}] {beat.description} (Emotion: {beat.emotion})\n"
                
            try:
                result_dict = self.llm.generate_json(prompt, schema)
            except Exception as e:
                logger.warning(f"LLM shot coverage extraction failed for scene {scene.scene_id}, falling back.")
                result_dict = {"beats": []}
                
            if not result_dict.get("beats"):
                logger.warning(f"LLM generated no shots, injecting fallback shot for scene {scene.scene_id}")
                result_dict = {"scene_style": "narrative", "beats": [{"shots": [{"purpose": "establishing", "emotion": "neutral", "importance": "high", "focus": "environment", "duration": 3.0}]}]}
                
            scene_style = result_dict.get("scene_style", "narrative").lower()
            shot_idx = 0
            for beat_idx, beat_cov in enumerate(result_dict.get("beats", [])):
                # Prefer positional correspondence to the scene's actual beats
                # over trusting the LLM to reproduce the exact beat_id string -
                # order is a much safer assumption than verbatim ID echo.
                if beat_idx < len(scene.beats):
                    beat_id = scene.beats[beat_idx].beat_id
                else:
                    beat_id = beat_cov.get("beat_id", "")
                for shot_data in beat_cov.get("shots", []):
                    shot_idx += 1
                    shot_id = f"shot_{scene_hash}_{shot_idx:03d}"
                    
                    # Ensure duration adheres loosely to style guide if LLM failed
                    duration = float(shot_data.get("duration", 2.0))
                    if "motion" in scene_style or "action" in scene_style:
                        duration = min(max(duration, 0.5), 2.5)
                    elif "montage" in scene_style:
                        duration = min(max(duration, 0.5), 2.0)
                        
                    shot = Shot(
                        shot_id=shot_id,
                        beat_id=beat_id,
                        purpose=shot_data.get("purpose", "mid"),
                        emotion=shot_data.get("emotion", "neutral"),
                        importance=shot_data.get("importance", "medium"),
                        focus=shot_data.get("focus", "character"),
                        duration=duration
                    )
                    all_shots.append(shot)
            
        manifest = ShotManifest(
            shots=all_shots,
            generator="ShotPlannerStage",
            generator_version="1.0.0",
            schema_version="2.0"
        )
        
        node = ExecutionNode(artifact=manifest, stage_name="ShotPlannerStage")
        
        return StageResult(
            artifact=manifest,
            execution_node=node,
            metrics={"total_shots": len(all_shots)},
            metadata={}
        )
