from core.pipeline.stage import CompilerStage, StageResult
from core.domain.assets.execution import ExecutionNode
from core.domain.scene.manifest import ShotManifest
from core.pipeline.context import PipelineContext
from typing import Any
import copy

class CameraPlannerStage(CompilerStage):
    def get_name(self) -> str:
        return "CameraPlannerStage"

    def get_providers(self) -> list:
        return []
        
    def inputs(self, context: PipelineContext) -> list[Any]:
        for node in context.execution_nodes:
            if isinstance(node.artifact, ShotManifest):
                return [node.artifact]
        return []
        
    def outputs(self) -> list[str]:
        return ["shot_manifest_with_camera"]
        
    def generator_signature(self) -> str:
        return f"{self.get_name()}_rule_engine_v1.0"
        
    def execute(self, context: PipelineContext) -> StageResult:
        shot_manifest = None
        for node in reversed(context.execution_nodes):
            if isinstance(node.artifact, ShotManifest):
                shot_manifest = node.artifact
                break
                
        if not shot_manifest:
            raise ValueError("No ShotManifest found in context.")
            
        enriched_manifest = copy.deepcopy(shot_manifest)
        # ShotManifest might not have generator fields, skip assignment.
        
        for shot in enriched_manifest.shots:
            purpose = shot.purpose.lower()
            emotion = shot.emotion.lower()
            focus = shot.focus.lower()
            
            shot.camera_type = "cinematic"
            
            # Distance / Framing
            if "establishing" in purpose or "wide" in purpose:
                shot.distance = "wide shot"
                shot.lens = "24mm"
            elif "closeup" in purpose or "close-up" in purpose or "reaction" in purpose:
                shot.distance = "close-up"
                shot.lens = "85mm"
            elif "insert" in purpose:
                shot.distance = "extreme close-up"
                shot.lens = "100mm macro"
            else:
                shot.distance = "medium shot"
                shot.lens = "50mm"
                
            # Angle
            if "intimidating" in emotion or "powerful" in emotion:
                shot.angle = "low angle"
            elif "vulnerable" in emotion or "weak" in emotion:
                shot.angle = "high angle"
            else:
                shot.angle = "eye-level"
                
            # Movement
            if "action" in emotion or "chaotic" in emotion:
                shot.movement = "handheld tracking"
            elif "establishing" in purpose:
                shot.movement = "slow dolly in"
            elif "reaction" in purpose:
                shot.movement = "static push-in"
            else:
                shot.movement = "static"
            
        node = ExecutionNode(artifact=enriched_manifest, stage_name="CameraPlannerStage")
        
        return StageResult(
            artifact=enriched_manifest,
            execution_node=node,
            metrics={"shots_planned": len(enriched_manifest.shots)},
            metadata={}
        )
