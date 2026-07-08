from core.pipeline.stage import PipelineStage, StageResult
from core.domain.assets.execution import ExecutionNode
from core.domain.prompt.ast import PromptManifest
from core.domain.scene.manifest import ShotManifest
import logging

logger = logging.getLogger(__name__)

class ValidatorStage(PipelineStage):
    def get_providers(self) -> list:
        return []
        
    def execute(self, context) -> StageResult:
        logger.info("Validating compiler manifests before GPU execution...")
        
        prompt_manifest = None
        shot_manifest = None
        
        for node in context.execution_nodes:
            if isinstance(node.artifact, PromptManifest):
                prompt_manifest = node.artifact
            elif isinstance(node.artifact, ShotManifest):
                shot_manifest = node.artifact
                
        if not prompt_manifest:
            raise ValueError("Validator: Missing PromptManifest.")
        if not shot_manifest:
            raise ValueError("Validator: Missing ShotManifest.")
            
        # Example Cheap Validation Rules
        if len(prompt_manifest.prompts) != len(shot_manifest.shots):
            raise ValueError("Validator: Prompt count does not match Shot count!")
            
        prompt_ids = set()
        for p in prompt_manifest.prompts:
            if p.prompt_id in prompt_ids:
                raise ValueError(f"Validator: Duplicate prompt_id {p.prompt_id}")
            prompt_ids.add(p.prompt_id)
            
            if p.ast.camera.type == "":
                raise ValueError(f"Validator: Prompt {p.prompt_id} is missing camera data.")
                
            if len(str(p.ast.subject)) > 500:
                raise ValueError(f"Validator: Prompt {p.prompt_id} subject is too long.")
                
        # Passed validation
        node = ExecutionNode(artifact=prompt_manifest, stage_name="ValidatorStage")
        
        return StageResult(
            artifact=prompt_manifest,
            execution_node=node,
            metrics={"valid_prompts": len(prompt_manifest.prompts)},
            metadata={"status": "passed"}
        )
