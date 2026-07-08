from core.pipeline.stage import PipelineStage, StageResult
from core.domain.assets.execution import ExecutionNode
from core.domain.prompt.ast import PromptManifest
import logging

logger = logging.getLogger(__name__)

class PromptValidatorStage(PipelineStage):
    def get_providers(self) -> list:
        return []

    def execute(self, context) -> StageResult:
        prompt_manifest = None
        for node in context.execution_nodes:
            if isinstance(node.artifact, PromptManifest) and getattr(node.artifact, "generator", "") == "PromptOptimizerStage":
                prompt_manifest = node.artifact
                break
                
        if not prompt_manifest:
            raise ValueError("PromptValidator: Missing Optimized PromptManifest.")
            
        for p in prompt_manifest.prompts:
            if not p.ast.subject.description and not p.ast.characters:
                raise ValueError(f"Prompt {p.prompt_id} is empty (no subject or characters).")
                
            if not p.ast.environment.location:
                logger.warning(f"Prompt {p.prompt_id} is missing a location.")
                
            if not p.ast.camera.type:
                logger.warning(f"Prompt {p.prompt_id} is missing camera data.")
                
            # Simulate checking banned words
            banned = ["nsfw", "gore"]
            for word in banned:
                if word in p.ast.subject.description.lower():
                    raise ValueError(f"Prompt {p.prompt_id} contains banned word: {word}")
                    
        node = ExecutionNode(artifact=prompt_manifest, stage_name="PromptValidatorStage")
        
        return StageResult(
            artifact=prompt_manifest,
            execution_node=node,
            metrics={"validated_prompts": len(prompt_manifest.prompts)},
            metadata={"status": "passed"}
        )
