from core.pipeline.stage import CompilerStage, StageResult
from core.domain.assets.execution import ExecutionNode
from core.domain.scene.manifest import SceneManifest, ShotManifest, CastMember
from core.pipeline.context import PipelineContext
from typing import Any
import copy
import logging

logger = logging.getLogger(__name__)

class CastPlannerStage(CompilerStage):
    def __init__(self, llm_provider=None):
        self.llm = llm_provider

    def get_name(self) -> str:
        return "CastPlannerStage"

    def get_providers(self) -> list:
        return [self.llm] if self.llm else []
        
    def inputs(self, context: PipelineContext) -> list[Any]:
        inputs = []
        for node in context.execution_nodes:
            if isinstance(node.artifact, (SceneManifest, ShotManifest)):
                inputs.append(node.artifact)
        return inputs
        
    def outputs(self) -> list[str]:
        return ["shot_manifest_with_cast"]
        
    def generator_signature(self) -> str:
        return f"{self.get_name()}_{type(self.llm).__name__ if self.llm else 'default'}_v1.0"

    def execute(self, context: PipelineContext) -> StageResult:
        if not self.llm:
            from plugins.local_llm import LocalLLMProvider
            self.llm = LocalLLMProvider()
            
        scene_manifest = None
        shot_manifest = None
        
        for node in context.execution_nodes:
            if isinstance(node.artifact, SceneManifest):
                scene_manifest = node.artifact
            elif isinstance(node.artifact, ShotManifest):
                shot_manifest = node.artifact
                
        if not scene_manifest or not shot_manifest:
            raise ValueError("CastPlanner requires both SceneManifest and ShotManifest.")
            
        enriched_manifest = copy.deepcopy(shot_manifest)
        # ShotManifest might not have generator fields, skip assignment.
        
        # Build mapping of scene characters
        scene_chars = {}
        for s in scene_manifest.scenes:
            scene_chars[s.scene_id] = s.characters

        # Group shots by scene
        shots_by_scene = {}
        for shot in enriched_manifest.shots:
            # shot_id format: shot_{scene_hash}_{idx}
            parts = shot.shot_id.split("_")
            if len(parts) >= 2:
                scene_hash = parts[1]
                if scene_hash not in shots_by_scene:
                    shots_by_scene[scene_hash] = []
                shots_by_scene[scene_hash].append(shot)
        
        hash_to_chars = {}
        for scene in scene_manifest.scenes:
            import hashlib
            h = hashlib.sha256(scene.scene_id.encode('utf-8')).hexdigest()[:8]
            hash_to_chars[h] = scene.characters
            
        schema = {
            "shots": [
                {
                    "shot_id": "string",
                    "cast": [
                        {
                            "character_id": "string",
                            "emotion": "string (e.g. sad, angry, neutral)",
                            "pose": "string (e.g. standing, kneeling, walking)",
                            "visibility": "string (e.g. foreground, background, hidden)",
                            "interaction": "string (e.g. looking at John, holding object)"
                        }
                    ]
                }
            ]
        }
            
        for scene_hash, shots in shots_by_scene.items():
            chars_available = hash_to_chars.get(scene_hash, [])
            if not chars_available:
                continue # No characters to cast
                
            prompt = f"""
You are a master Casting Director and Film Editor. For each shot in the sequence, determine the state of the characters present.

Characters available in this scene: {', '.join(chars_available)}

Shots:
"""
            for shot in shots:
                prompt += f"- [Shot {shot.shot_id}] Purpose: {shot.purpose}, Emotion: {shot.emotion}, Focus: {shot.focus}\n"
                
            prompt += "\nOutput the cast state for each shot using the exact shot_id."
            
            try:
                result_dict = self.llm.generate_json(prompt, schema)
                cast_map = {}
                for s_data in result_dict.get("shots", []):
                    cast_map[s_data.get("shot_id")] = s_data.get("cast", [])
                    
                for shot in shots:
                    shot_cast = cast_map.get(shot.shot_id, [])
                    shot.cast = [CastMember(**c) for c in shot_cast]
            except Exception as e:
                logger.warning(f"CastPlanner LLM failed for scene {scene_hash}: {e}")
        
        node = ExecutionNode(artifact=enriched_manifest, stage_name="CastPlannerStage")
        
        return StageResult(
            artifact=enriched_manifest,
            execution_node=node,
            metrics={"shots_cast": len(enriched_manifest.shots)},
            metadata={}
        )
