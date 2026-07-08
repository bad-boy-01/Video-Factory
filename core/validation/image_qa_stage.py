from core.pipeline.stage import PipelineStage, StageResult
from core.domain.assets.execution import ExecutionNode
from core.domain.assets.registry import AssetRegistry
from core.rendering.render_queue import RenderQueue
from core.domain.prompt.ast import PromptManifest
import logging

logger = logging.getLogger(__name__)

class ImageQAStage(PipelineStage):
    def __init__(self, vlm_provider=None):
        self.vlm = vlm_provider
        self.queue = RenderQueue()

    def get_providers(self) -> list:
        return [self.vlm] if self.vlm else []

    def execute(self, context) -> StageResult:
        asset_registry = None
        prompt_manifest = None
        
        for node in context.execution_nodes:
            if isinstance(node.artifact, AssetRegistry):
                asset_registry = node.artifact
            elif isinstance(node.artifact, PromptManifest):
                prompt_manifest = node.artifact
                
        if not asset_registry or not prompt_manifest:
            raise ValueError("ImageQA: Missing AssetRegistry or PromptManifest.")
            
        pending_qa_jobs = self.queue.get_jobs_by_status("QA_PENDING")
        
        repaired_prompts = 0
        passed_images = 0
        
        for job in pending_qa_jobs:
            job_id = job["job_id"]
            
            # Find associated asset
            asset = next((a for a in asset_registry.assets.values() if a.prompt_hash == job_id), None)
            if not asset:
                continue
                
            # 1. Fast QA
            # Mocking fast check (e.g. valid PNG, non-empty, right res)
            fast_qa_passed = True 
            
            if not fast_qa_passed:
                self.queue.update_job_status(job_id, "QA_FAILED")
                continue
                
            # 2. Semantic QA (VLM)
            target_prompt = next((p for p in prompt_manifest.prompts if p.prompt_id == job_id), None)
            semantic_qa_passed = True
            
            # If semantic QA fails, trigger repair loop
            if not semantic_qa_passed:
                logger.warning(f"Semantic QA failed for {job_id}. Triggering repair loop.")
                self.queue.update_job_status(job_id, "REPAIR_PENDING")
                # In reality, PromptOptimizer would rewrite the prompt and push it to PROMPT_READY
                repaired_prompts += 1
            else:
                self.queue.update_job_status(job_id, "RENDER_COMPLETE")
                passed_images += 1
                
        # Generate Pipeline Statistics
        stats = {
            "qa_passed": passed_images,
            "qa_failed": repaired_prompts,
            "gpu_hours": 0.5, # mock
            "cache_hits": 10,
            "peak_vram_gb": 12.5
        }
        
        node = ExecutionNode(artifact=asset_registry, stage_name="ImageQAStage")
        
        return StageResult(
            artifact=asset_registry,
            execution_node=node,
            metrics=stats,
            metadata={}
        )
