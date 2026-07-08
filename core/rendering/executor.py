import logging
from typing import List
from core.pipeline.context import PipelineContext
from core.pipeline.stage import PipelineStage
from core.contracts.engine import ContractEngine
from core.contracts.router import ContractRouter

logger = logging.getLogger(__name__)

class CompilerExecutor:
    """
    Executor that runs stages, guarded by the Generative Contract System and Incremental Cache.
    """
    def __init__(self, stages: List[PipelineStage], contract_router: ContractRouter, max_retries: int = 2):
        self.stages = stages
        self.router = contract_router
        self.max_retries = max_retries
        
        from core.pipeline.reducer import ContextReducer
        self.reducer = ContextReducer()

    def run(self, context: PipelineContext) -> PipelineContext:
        import time
        logger.info("Starting Compiler Executor")
        
        current_context = context
        timeline_logs = []
        pipeline_start_time = time.time()
        
        for i, stage in enumerate(self.stages):
            retries = 0
            stage_name = stage.__class__.__name__
            stage_start_time = time.time()

            # Note: Provider loading/unloading is now managed externally by ResourceSession.

            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
            except ImportError:
                pass

            # Incremental Cache Check (DAG-aware)
            cache_hit = False
            candidate_result = None
            invalidation_reasons = []
            
            # The executor orchestrates the hashing instead of the stage doing it manually
            if hasattr(stage, "inputs") and hasattr(stage, "outputs"):
                inputs = stage.inputs(current_context)
                outputs = stage.outputs()
                
                # Compute logical dependency hash for this stage
                import hashlib
                import json
                dep_parts = []
                for inp in inputs:
                    if hasattr(inp, "metadata"):
                        dep_parts.append(getattr(inp.metadata, "fingerprint", str(inp.metadata)))
                    elif hasattr(inp, "fingerprint"):
                        dep_parts.append(inp.fingerprint)
                    else:
                        dep_parts.append(str(inp)) # Fallback
                
                current_dep_hash = hashlib.sha256("_".join(dep_parts).encode()).hexdigest()
                current_gen_sig = stage.generator_signature()
                
                # Check cache for outputs
                if hasattr(current_context, "workspace"):
                    for out_name in outputs:
                        cache_file = current_context.workspace.manifests_dir / f"{out_name}.json"
                        if cache_file.exists():
                            with open(cache_file, "r", encoding="utf-8") as f:
                                cached_data = json.load(f)
                                
                            metadata = cached_data.get("metadata", {})
                            cached_dep_hash = metadata.get("dependency_hash")
                            cached_gen_sig = metadata.get("generator_signature")
                            cached_schema = metadata.get("schema_version")
                            
                            # Cache validation rules
                            if cached_schema != "1.0":
                                invalidation_reasons.append("schema version changed")
                            if cached_gen_sig != current_gen_sig:
                                invalidation_reasons.append("generator signature changed")
                            if cached_dep_hash != current_dep_hash:
                                invalidation_reasons.append("dependency hash changed")
                                
                            if not invalidation_reasons:
                                logger.info(f"[{stage_name}] Cache HIT")
                                cache_hit = True
                                
                                # Convert dict back to correct domain model if needed.
                                # For now we rely on the reducer or stage to handle the raw dict or we rebuild the artifact.
                                # We can't automatically infer the class without a registry, so we let the stage instantiate it if we have `load_cached_artifact`
                                # Or we expect the stage to return it.
                                if hasattr(stage, "load_cached_artifact"):
                                    cached_artifact = stage.load_cached_artifact(current_context.workspace)
                                    from core.pipeline.stage import StageResult
                                    from core.domain.assets.execution import ExecutionNode
                                    node = ExecutionNode(artifact=cached_artifact, stage_name=stage_name)
                                    candidate_result = StageResult(
                                        artifact=cached_artifact,
                                        execution_node=node,
                                        metrics={"cache_hit": True},
                                        metadata={}
                                    )
                                else:
                                    logger.warning(f"[{stage_name}] Hit cache but stage lacks load_cached_artifact. Rebuilding.")
                                    cache_hit = False
                            else:
                                logger.info(f"[{stage_name}] Cache MISS")
                                logger.info(f"Reason: \n  • " + "\n  • ".join(invalidation_reasons))
                                logger.info("Rebuilding...")

            while not cache_hit:
                logger.info(f"Starting stage: {stage_name} (Attempt {retries + 1})")
                
                try:
                    candidate_result = stage.execute(current_context)
                    candidate_result.metrics["cache_hit"] = False
                    
                    contracts = self.router.get_contracts(stage_name)
                    engine = ContractEngine(contracts)
                    engine.run(current_context, candidate_result.artifact)
                    break  # PASS
                except Exception as e:
                    retries += 1
                    logger.warning(f"[RETRY] {stage_name}: Failed ({retries}/{self.max_retries}) - {str(e)}", exc_info=True)
                    if retries >= self.max_retries:
                        logger.error(f"[FATAL] {stage_name} exhausted all retries.", exc_info=True)
                        raise e
            
            # Reduce context whether from cache or fresh execution
            if candidate_result:
                current_context = self.reducer.reduce(current_context, candidate_result)

            stage_duration = time.time() - stage_start_time
            peak_alloc = 0.0
            peak_res = 0.0
            cur_alloc = 0.0
            cur_res = 0.0
            try:
                import torch
                if torch.cuda.is_available():
                    peak_alloc = torch.cuda.max_memory_allocated() / (1024**3)
                    peak_res = torch.cuda.max_memory_reserved() / (1024**3)
                    cur_alloc = torch.cuda.memory_allocated() / (1024**3)
                    cur_res = torch.cuda.memory_reserved() / (1024**3)
            except ImportError:
                pass
            
            cache_str = "HIT" if cache_hit else "MISS"
            
            timeline_logs.append(
                f"[{i+1}/{len(self.stages)}] {stage_name:<20} {stage_duration:.2f} s\n"
                f"        Cache: {cache_str}\n"
                f"        Alloc: {cur_alloc:.2f} GB (Peak: {peak_alloc:.2f} GB)\n"
                f"        Reserv: {cur_res:.2f} GB (Peak: {peak_res:.2f} GB)\n"
                f"        Contracts: PASS\n"
                f"        Retry: {retries}"
            )

        # Generate DAG
        import os
        from pathlib import Path
        reports_dir = Path(getattr(current_context.workspace, "base_dir", "workspace")) / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        dot_path = reports_dir / "compiler_dag.dot"
        
        dot_lines = ["digraph CompilerDAG {", "  rankdir=TB;", "  node [shape=box, style=filled, fillcolor=lightgray];"]
        for stage in self.stages:
            if hasattr(stage, "get_name") and hasattr(stage, "outputs"):
                stage_name = stage.get_name()
                dot_lines.append(f'  "{stage_name}" [shape=ellipse, fillcolor=lightblue];')
                if hasattr(stage, "inputs"):
                    # We just use string representation for the DAG
                    for inp in stage.inputs(current_context):
                        if hasattr(inp, "metadata") and getattr(inp.metadata, "artifact_type", None):
                            dot_lines.append(f'  "{inp.metadata.artifact_type}" -> "{stage_name}";')
                for out in stage.outputs():
                    dot_lines.append(f'  "{stage_name}" -> "{out}";')
        dot_lines.append("}")
        
        with open(dot_path, "w") as f:
            f.write("\n".join(dot_lines))
            
        logger.info(f"Compiler DAG generated at {dot_path}")
        
        logger.info("\n" + "="*40 + "\nSTAGE TIMELINE\n" + "="*40)
        for log in timeline_logs:
            logger.info(log)
            
        total_time = time.time() - pipeline_start_time
        logger.info(f"\nTotal Time: {total_time:.2f} s")
        logger.info("="*40 + "\n")
        
        return current_context
