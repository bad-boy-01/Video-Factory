from core.pipeline.stage import PipelineStage, StageResult
from core.domain.assets.execution import ExecutionNode
from core.domain.prompt.ast import PromptManifest
from core.domain.assets.registry import AssetRegistry, Asset
from core.rendering.render_queue import RenderQueue
from core.rendering.render_graph import RenderNode, RenderArtifact, RenderGraph
import logging
import os
import json
import math
import datetime
from typing import Dict, Any

logger = logging.getLogger(__name__)

class CompilePromptNode(RenderNode):
    def __init__(self, provider_compiler):
        self.compiler = provider_compiler
        
    def get_name(self) -> str:
        return "CompilePromptNode"
        
    def execute(self, inputs: Dict[str, RenderArtifact], config: Dict[str, Any]) -> Dict[str, RenderArtifact]:
        plan = inputs["RENDER_PLAN"].data
        request = self.compiler.compile_plan(plan)
        return {"PROVIDER_REQUEST": RenderArtifact(kind="PROVIDER_REQUEST", data=request)}

class GenerateNode(RenderNode):
    def __init__(self, provider):
        self.provider = provider
        
    def get_name(self) -> str:
        return "GenerateNode"
        
    def execute(self, inputs: Dict[str, RenderArtifact], config: Dict[str, Any]) -> Dict[str, RenderArtifact]:
        request = inputs["PROVIDER_REQUEST"].data
        image = self.provider.generate(request)
        return {"IMAGE": RenderArtifact(kind="IMAGE", data=image)}

class SaveNode(RenderNode):
    def get_name(self) -> str:
        return "SaveNode"
        
    def execute(self, inputs: Dict[str, RenderArtifact], config: Dict[str, Any]) -> Dict[str, RenderArtifact]:
        image = inputs["IMAGE"].data
        shot_dir = config["shot_dir"]
        plan = inputs["RENDER_PLAN"].data
        output_path = str(shot_dir / f"{plan.shot_id}.png")
        image.save(output_path)
        return {"SAVED_IMAGE_PATH": RenderArtifact(kind="PATH", data=output_path)}

class DiffusionRendererStage(PipelineStage):
    def __init__(self, provider_compiler=None, diffusion_provider=None, render_options=None):
        self.compiler = provider_compiler
        self.diffusion = diffusion_provider
        self.render_options = render_options or {}

    def get_providers(self) -> list:
        return [self.diffusion] if self.diffusion else []

    def execute(self, context) -> StageResult:
        if not self.compiler:
            from plugins.local_diffusion import DiffusersCompiler
            self.compiler = DiffusersCompiler()
            
        if not self.diffusion:
            from plugins.local_diffusion import DiffusersProvider
            self.diffusion = DiffusersProvider()
            
        logger.info("Executing DiffusionRendererStage via RenderGraph...")
        
        # Build the graph
        graph = RenderGraph()
        graph.add_node(CompilePromptNode(self.compiler))
        graph.add_node(GenerateNode(self.diffusion))
        graph.add_node(SaveNode())
        
        prompt_manifest = None
        for node in context.execution_nodes:
            if isinstance(node.artifact, PromptManifest):
                prompt_manifest = node.artifact
                break
                
        if not prompt_manifest:
            raise ValueError("DiffusionRendererStage: Missing PromptManifest.")
            
        from core.domain.prompt.render_plan import RenderPlan, LogicalRenderPlan, PhysicalRenderPlan
        
        jobs_processed = 0
        for entry in prompt_manifest.prompts:
            # PromptBuilderStage stores the fully assembled positive prompt in
            # ast.subject.description and negative tags in ast.negative.tags.
            full_prompt = entry.ast.subject.description
            full_negative = ", ".join(entry.ast.negative.tags)

            # Extract style_name from VisualScene if present
            style_name = ""
            if entry.visual_scene and entry.visual_scene.style:
                # Convention: prompt_builder stores style key in color_grade when set
                pass
            # Fall back to shot style on the first available cast shot
            # (style_name is set on Shot.style by DirectorPlannerStage)

            plan = RenderPlan(
                shot_id=entry.shot_id,
                logical=LogicalRenderPlan(
                    subject=entry.ast.subject.description,
                    framing=entry.ast.camera.distance,
                    emphasis="",
                    mood=entry.ast.mood.mood,
                    full_prompt=full_prompt,
                    full_negative=full_negative,
                    style_name=style_name,
                ),
                physical=PhysicalRenderPlan(
                    width=entry.ast.technical.width,
                    height=entry.ast.technical.height,
                    steps=entry.ast.technical.steps,
                    cfg=entry.ast.technical.cfg,
                    seed=entry.seed
                )
            )

            inputs = {"RENDER_PLAN": RenderArtifact(kind="RENDER_PLAN", data=plan)}
            config = {"shot_dir": context.workspace.outputs_dir}
            graph.execute(inputs, config)
            jobs_processed += 1
            
        registry = context.registry
        node = ExecutionNode(artifact=registry, stage_name="DiffusionRendererStage")
        
        return StageResult(
            artifact=registry,
            execution_node=node,
            metrics={"jobs_processed": jobs_processed},
            metadata={}
        )
