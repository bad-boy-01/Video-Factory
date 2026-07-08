from core.pipeline.stage import CompilerStage, StageResult
from core.domain.assets.execution import ExecutionNode
from core.domain.scene.manifest import SceneManifest, SceneState
from core.domain.story.bible import StoryBible
from core.pipeline.context import PipelineContext
from typing import Any
import copy
import logging

logger = logging.getLogger(__name__)

class VisualContinuityStage(CompilerStage):
    def __init__(self, llm_provider=None):
        self.llm = llm_provider

    def get_name(self) -> str:
        return "VisualContinuityStage"

    def get_providers(self) -> list:
        return [self.llm] if self.llm else []
        
    def inputs(self, context: PipelineContext) -> list[Any]:
        inputs = []
        for node in context.execution_nodes:
            if isinstance(node.artifact, (SceneManifest, StoryBible)):
                inputs.append(node.artifact)
        return inputs
        
    def outputs(self) -> list[str]:
        return ["scene_manifest_with_continuity"]
        
    def generator_signature(self) -> str:
        return f"{self.get_name()}_{type(self.llm).__name__ if self.llm else 'default'}_v1.0"

    def execute(self, context: PipelineContext) -> StageResult:
        if not self.llm:
            from plugins.local_llm import LocalLLMProvider
            self.llm = LocalLLMProvider()
            
        scene_manifest = None
        bible = None
        
        for node in context.execution_nodes:
            if isinstance(node.artifact, SceneManifest):
                scene_manifest = node.artifact
            elif isinstance(node.artifact, StoryBible):
                bible = node.artifact
                
        if not scene_manifest or not bible:
            raise ValueError("VisualContinuityPlanner requires both SceneManifest and StoryBible.")
            
        enriched_manifest = copy.deepcopy(scene_manifest)
        enriched_manifest.generator = "VisualContinuityStage"
        enriched_manifest.generator_version = "1.0.0"

        schema = {
            "time": "string (e.g. night, sunset, noon)",
            "season": "string",
            "weather": "string",
            "lighting": "string",
            "palette": "string",
            "wardrobe_locks": {"type": "object", "additionalProperties": {"type": "string"}},
            "hair_state": {"type": "object", "additionalProperties": {"type": "string"}},
            "damage_state": {"type": "object", "additionalProperties": {"type": "string"}},
            "prop_state": {"type": "object", "additionalProperties": {"type": "string"}},
            "vehicle_state": {"type": "object", "additionalProperties": {"type": "string"}},
            "environment_state": "string"
        }
        
        for scene in enriched_manifest.scenes:
            loc_data = bible.locations.get(scene.location)
            loc_str = f"Location: {scene.location}"
            if loc_data:
                loc_str += f" (Weather: {loc_data.weather_defaults}, Time: {loc_data.time_defaults})"
                
            prompt = f"""
You are a master Continuity Director. Lock the visual state for this scene.
Scene ID: {scene.scene_id}
{loc_str}
Emotion: {scene.emotion}
Characters present: {', '.join(scene.characters)}

For wardrobe_locks, assign one of the character's known wardrobe names if applicable.
Return the exact JSON structure for SceneState.
"""
            try:
                result_dict = self.llm.generate_json(prompt, schema)
                scene.state = SceneState(
                    time=result_dict.get("time", ""),
                    season=result_dict.get("season", ""),
                    weather=result_dict.get("weather", ""),
                    lighting=result_dict.get("lighting", ""),
                    palette=result_dict.get("palette", ""),
                    wardrobe_locks=result_dict.get("wardrobe_locks", {}),
                    hair_state=result_dict.get("hair_state", {}),
                    damage_state=result_dict.get("damage_state", {}),
                    prop_state=result_dict.get("prop_state", {}),
                    vehicle_state=result_dict.get("vehicle_state", {}),
                    environment_state=result_dict.get("environment_state", "")
                )
            except Exception as e:
                logger.warning(f"LLM failed to generate continuity state for scene {scene.scene_id}")
                scene.state = SceneState()
        
        node = ExecutionNode(artifact=enriched_manifest, stage_name="VisualContinuityStage")
        
        return StageResult(
            artifact=enriched_manifest,
            execution_node=node,
            metrics={"scenes_locked": len(enriched_manifest.scenes)},
            metadata={}
        )
