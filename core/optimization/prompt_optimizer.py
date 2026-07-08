from core.pipeline.stage import PipelineStage, StageResult
from core.domain.assets.execution import ExecutionNode
from core.domain.prompt.ast import PromptManifest
import copy

class ModelOptimizer:
    def optimize(self, manifest: PromptManifest) -> PromptManifest:
        raise NotImplementedError

class SDXLOptimizer(ModelOptimizer):
    def optimize(self, manifest: PromptManifest) -> PromptManifest:
        optimized = copy.deepcopy(manifest)
        for p in optimized.prompts:
            p.model_target = "sdxl"
            
            # SDXL prefers quality tags first, natural language description, then camera/lighting
            # Remove duplicate adjectives from quality tags
            unique_quality = list(dict.fromkeys(p.ast.quality.tags))
            p.ast.quality.tags = [q for q in unique_quality if q]
            
            # SDXL resolution is usually 1024x1024
            p.ast.technical.width = 1024
            p.ast.technical.height = 1024
            
        return optimized

class PromptOptimizerStage(PipelineStage):
    def __init__(self, target_model: str = "sdxl"):
        self.target_model = target_model
        
    def get_providers(self) -> list:
        return []

    def execute(self, context) -> StageResult:
        prompt_manifest = None
        for node in context.execution_nodes:
            if isinstance(node.artifact, PromptManifest):
                prompt_manifest = node.artifact
                break
                
        if not prompt_manifest:
            raise ValueError("PromptOptimizer: Missing PromptManifest.")
            
        optimizer = None
        if self.target_model == "sdxl":
            optimizer = SDXLOptimizer()
        else:
            raise ValueError(f"Unknown target model: {self.target_model}")
            
        optimized_manifest = optimizer.optimize(prompt_manifest)
        optimized_manifest.generator = "PromptOptimizerStage"
        
        node = ExecutionNode(artifact=optimized_manifest, stage_name="PromptOptimizerStage")
        
        return StageResult(
            artifact=optimized_manifest,
            execution_node=node,
            metrics={"optimized_prompts": len(optimized_manifest.prompts)},
            metadata={"target_model": self.target_model}
        )
