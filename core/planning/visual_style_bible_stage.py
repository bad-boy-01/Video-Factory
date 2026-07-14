"""
VisualStyleBibleStage — creates the hierarchical VisualStyleBible.

Deterministic Stage: Reads StoryBible + DirectorManifest.
Constructs the GlobalStyle based on DirectorManifest's target_visual_style.
Can optionally populate SceneStyles if ambient_conditions exist in the SceneGraph,
but mostly it just sets up the immutable style anchor for the project.
"""
from __future__ import annotations

import logging
from typing import Any

from core.pipeline.stage import CompilerStage, StageResult
from core.domain.assets.execution import ExecutionNode
from core.domain.story.director_manifest import DirectorManifest
from core.domain.style.visual_style_bible import (
    VisualStyleBible, GlobalStyle, SceneStyle
)
from core.domain.scene.graph import SceneGraphManifest
from core.pipeline.context import PipelineContext

logger = logging.getLogger(__name__)

# Predefined base styles
BASE_STYLES = {
    "korean_manhwa_cinematic": GlobalStyle(
        style_name="korean_manhwa_cinematic",
        line_art="bold clean lineart, thick outlines, sharp crisp edges, Korean manhwa style",
        shading="cel-shaded with soft gradient shadows, ambient occlusion, flat color blocks",
        color_palette="highly saturated vibrant palette, strong local color contrast",
        face_style="manhwa facial proportions, large expressive eyes, sharp jawline, smooth skin",
        background_style="semi-detailed backgrounds, architectural precision, depth blur behind subject",
        lighting="dramatic three-point lighting, strong rim light, deep cast shadows",
    ),
    "dark_fantasy_cinematic": GlobalStyle(
        style_name="dark_fantasy_cinematic",
        line_art="intricate linework, crosshatching, gritty texture",
        shading="heavy chiaroscuro, deep blacks, painterly rendering",
        color_palette="desaturated, muted earthy tones, stark crimson accents",
        face_style="realistic proportions, weathered skin, intense expressions",
        background_style="gothic architecture, heavy fog, looming silhouettes",
        lighting="high contrast, low key lighting, harsh shadows",
    ),
    "slice_of_life_webtoon": GlobalStyle(
        style_name="slice_of_life_webtoon",
        line_art="delicate thin lineart, soft edges, modern webtoon style",
        shading="soft pastel shading, bright ambient light, minimal harsh shadows",
        color_palette="pastel colors, bright airy tones, warm whites",
        face_style="soft facial features, expressive large eyes, gentle blushing",
        background_style="bright interiors, detailed everyday objects, sunny windows",
        lighting="soft diffused daylight, bright airy atmosphere",
    )
}

class VisualStyleBibleStage(CompilerStage):
    """
    Deterministic Stage: Constructs the VisualStyleBible.
    """

    def get_name(self) -> str:
        return "VisualStyleBibleStage"

    def get_providers(self) -> list:
        return []

    def inputs(self, context: PipelineContext) -> list[Any]:
        results = []
        for node in context.execution_nodes:
            if isinstance(node.artifact, DirectorManifest):
                results.append(node.artifact)
        return results

    def outputs(self) -> list[str]:
        return ["style_bible"]

    def generator_signature(self) -> str:
        return f"{self.get_name()}_v1.0"

    def execute(self, context: PipelineContext) -> StageResult:
        director: DirectorManifest | None = None
        scene_graph_manifest: SceneGraphManifest | None = None

        for node in context.execution_nodes:
            if isinstance(node.artifact, DirectorManifest):
                director = node.artifact
            elif isinstance(node.artifact, SceneGraphManifest):
                scene_graph_manifest = node.artifact

        if not director:
            logger.warning("No DirectorManifest found. Using default manhwa style.")
            director = DirectorManifest()

        target_style = director.facts.target_visual_style
        global_style = BASE_STYLES.get(target_style, BASE_STYLES["korean_manhwa_cinematic"])
        
        # Apply color grade from policy
        if director.policy.color_grade_global:
            # We append it to the color_palette
            global_style.color_palette += f", {director.policy.color_grade_global.replace('_', ' ')} color grade"

        bible = VisualStyleBible(global_style=global_style)

        # Populate SceneStyles from ambient conditions if we have the scene graphs
        if scene_graph_manifest:
            for scene_id, graph in scene_graph_manifest.graphs.items():
                if graph.ambient_conditions:
                    ss = SceneStyle(scene_id=scene_id)
                    time = graph.ambient_conditions.get("time", "")
                    weather = graph.ambient_conditions.get("weather", "")
                    lighting = graph.ambient_conditions.get("lighting", "")
                    
                    if weather:
                        ss.weather_tokens = f"{weather} weather, atmospheric {weather}"
                    if time:
                        ss.time_of_day_tokens = f"{time}time, {time} sky"
                    if lighting:
                        ss.lighting = lighting
                    
                    bible.scene_styles[scene_id] = ss

        node = ExecutionNode(artifact=bible, stage_name=self.get_name())

        return StageResult(
            artifact=bible,
            execution_node=node,
            metrics={"style": target_style, "scene_styles_populated": len(bible.scene_styles)},
            metadata={},
        )
