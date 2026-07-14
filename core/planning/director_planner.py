from core.pipeline.stage import CompilerStage, StageResult
from core.domain.assets.execution import ExecutionNode
from core.domain.scene.manifest import SceneManifest, ShotManifest, StyleProfile, StyleStack
from core.pipeline.context import PipelineContext
from typing import Any
import copy
import logging

logger = logging.getLogger(__name__)

# Hardcoded director style rules
DIRECTOR_STYLES = {
    "villeneuve": StyleProfile(
        name="Villeneuve",
        camera={"angle": "eye-level", "distance": "wide"},
        lighting={"style": "soft diffuse, high contrast silhouettes"},
        composition={"style": "symmetrical, vast scale"},
        color={"palette": "monochromatic, stark"},
        editing={"pace": "slow"},
        motion={"movement": "slow push-in"},
        lens={"lens": "35mm anamorphic"}
    ),
    "webtoon": StyleProfile(
        name="Webtoon",
        camera={"angle": "dynamic, extreme high/low", "distance": "close-up"},
        lighting={"style": "flat, bright, cel-shaded"},
        composition={"style": "vertical scroll flow, dynamic panels"},
        color={"palette": "highly saturated, vibrant"},
        editing={"pace": "fast, punchy"},
        motion={"movement": "speed lines, fast pan"},
        lens={"lens": "14mm wide"}
    )
}

class DirectorPlannerStage(CompilerStage):
    def __init__(self, llm_provider=None):
        self.llm = llm_provider

    def get_name(self) -> str:
        return "DirectorPlannerStage"

    def get_providers(self) -> list:
        return [self.llm] if self.llm else []
        
    def inputs(self, context: PipelineContext) -> list[Any]:
        inputs = []
        for node in context.execution_nodes:
            if isinstance(node.artifact, (SceneManifest, ShotManifest)):
                inputs.append(node.artifact)
        return inputs
        
    def outputs(self) -> list[str]:
        return ["shot_manifest_with_style"]
        
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
            raise ValueError("DirectorPlanner requires both SceneManifest and ShotManifest.")
            
        enriched_manifest = copy.deepcopy(shot_manifest)
        # NOTE: Provenance is tracked via ExecutionNode.stage_name, not via
        # fields on the domain model (ShotManifest has no generator field).

        schema = {
            "project_style_name": "string (e.g. villeneuve, webtoon)"
        }
        
        # Ask the LLM to choose a global director style for the project
        prompt = "Based on the scenes, choose the most appropriate director style for this project from the following list: [villeneuve, webtoon]"
        
        try:
            result_dict = self.llm.generate_json(prompt, schema)
            project_style_name = result_dict.get("project_style_name", "villeneuve").lower()
            if project_style_name not in DIRECTOR_STYLES:
                project_style_name = "villeneuve"
        except Exception as e:
            logger.warning("LLM failed to select style, defaulting to Villeneuve")
            project_style_name = "villeneuve"
            
        # Instead of storing the style on each shot, we can just attach it to the Shot class directly.
        for shot in enriched_manifest.shots:
            shot.style = project_style_name 
        
        node = ExecutionNode(artifact=enriched_manifest, stage_name="DirectorPlannerStage")
        
        return StageResult(
            artifact=enriched_manifest,
            execution_node=node,
            metrics={"style_selected": project_style_name},
            metadata={}
        )
