from core.pipeline.stage import PipelineStage, StageResult
from core.domain.assets.execution import ExecutionNode
from core.domain.scene.manifest import SceneManifest
from core.domain.scene.memory import SceneMemory
from core.domain.base import DomainModel
import logging

logger = logging.getLogger(__name__)

class SceneMemoryManifest(DomainModel):
    memories: list[SceneMemory] = []

class StoryContinuityEngineStage(PipelineStage):
    def __init__(self, llm_provider=None):
        self.llm = llm_provider
        
    def get_providers(self) -> list:
        return [self.llm] if self.llm else []

    def execute(self, context) -> StageResult:
        scene_manifest = None
        for node in context.execution_nodes:
            if isinstance(node.artifact, SceneManifest):
                scene_manifest = node.artifact
                break
                
        if not scene_manifest:
            raise ValueError("StoryContinuityEngine: Missing SceneManifest.")
            
        memories = []
        last_weather = "Clear"
        last_time = "Morning"
        last_location = "Unknown"
        
        for scene in scene_manifest.scenes:
            # Here an LLM would normally analyze continuity based on the Scene text.
            # We carry forward state for M3 scaffolding.
            current_weather = last_weather
            current_time = last_time
            current_location = scene.location if scene.location else last_location
            
            memory = SceneMemory(
                scene_id=scene.scene_id,
                current_location=current_location,
                current_weather=current_weather,
                time_of_day=current_time,
                active_characters=scene.characters,
                mood=scene.emotion
            )
            memories.append(memory)
            
            # Update state for next scene
            last_weather = current_weather
            last_time = current_time
            last_location = current_location
            
        manifest = SceneMemoryManifest(memories=memories, generator="StoryContinuityEngineStage")
        node = ExecutionNode(artifact=manifest, stage_name="StoryContinuityEngineStage")
        
        return StageResult(
            artifact=manifest,
            execution_node=node,
            metrics={"memories_generated": len(memories)},
            metadata={}
        )
