"""
PromptBuilderStage — delegates to the modular PromptCompiler.

Replaces the old monolithic prompt builder.
This stage collects all manifests (StoryBible, DirectorManifest, VisualStyleBible,
StoryboardManifest, SceneManifest, ShotManifest) and invokes the PromptCompiler
to generate the final positive and negative prompts for each shot.
"""
from __future__ import annotations

import logging
import hashlib
from typing import Any, Dict

from core.pipeline.stage import CompilerStage, StageResult
from core.domain.assets.execution import ExecutionNode
from core.domain.prompt.ast import (
    PromptManifest, PromptManifestEntry, PromptAST, CameraAST,
    SubjectAST, EnvironmentAST, LightingAST, CompositionAST, QualityAST, NegativeAST, CharacterAST
)
from core.domain.scene.manifest import SceneManifest, ShotManifest
from core.domain.story.bible import StoryBible
from core.domain.story.director_manifest import DirectorManifest
from core.domain.style.visual_style_bible import VisualStyleBible
from core.domain.scene.storyboard import StoryboardManifest, MicroBeat
from core.optimization.prompt_compiler import PromptCompiler
from core.pipeline.context import PipelineContext

logger = logging.getLogger(__name__)


class PromptBuilderStage(CompilerStage):
    def get_name(self) -> str:
        return "PromptBuilderStage"

    def get_providers(self) -> list:
        return []

    def inputs(self, context: PipelineContext) -> list[Any]:
        inputs = []
        for node in context.execution_nodes:
            if isinstance(node.artifact, (
                SceneManifest, ShotManifest, StoryBible, DirectorManifest,
                VisualStyleBible, StoryboardManifest
            )):
                inputs.append(node.artifact)
        return inputs

    def outputs(self) -> list[str]:
        return ["prompt_manifest"]

    def generator_signature(self) -> str:
        return f"{self.get_name()}_compiler_v4.0"

    def execute(self, context: PipelineContext) -> StageResult:
        shot_manifest = None
        scene_manifest = None
        bible = None
        director = None
        style_bible = None
        storyboard = None
        comp_manifest = None

        for node in reversed(context.execution_nodes):
            if shot_manifest is None and isinstance(node.artifact, ShotManifest):
                shot_manifest = node.artifact
            elif scene_manifest is None and isinstance(node.artifact, SceneManifest):
                scene_manifest = node.artifact
            elif bible is None and isinstance(node.artifact, StoryBible):
                bible = node.artifact
            elif director is None and isinstance(node.artifact, DirectorManifest):
                director = node.artifact
            elif style_bible is None and isinstance(node.artifact, VisualStyleBible):
                style_bible = node.artifact
            elif storyboard is None and isinstance(node.artifact, StoryboardManifest):
                storyboard = node.artifact
            elif comp_manifest is None and getattr(node.artifact, '__class__', None).__name__ == 'CompositionManifest':
                comp_manifest = node.artifact

        if not shot_manifest:
            raise ValueError("PromptBuilder requires ShotManifest.")
        if not scene_manifest:
            raise ValueError("PromptBuilder requires SceneManifest.")

        # Instantiate modular PromptCompiler
        compiler = PromptCompiler(
            style_bible=style_bible,
            director=director,
            bible=bible,
        )

        # Build fast lookups
        scene_lookup = {s.scene_id: s for s in scene_manifest.scenes}
        
        # micro_beat lookup by shot_id
        # Recall that ShotPlanner assigns shot_id = f"shot_{mb.micro_beat_id}"
        micro_beat_lookup: Dict[str, MicroBeat] = {}
        if storyboard:
            for card in storyboard.cards:
                for mb in card.micro_beats:
                    # In ShotPlanner we did: shot_id = f"shot_{mb.micro_beat_id}"
                    # Just in case, let's index both forms.
                    micro_beat_lookup[mb.micro_beat_id] = mb
                    micro_beat_lookup[f"shot_{mb.micro_beat_id}"] = mb

            # Build fast lookup for Composition
            comp_lookup = {}
            if comp_manifest:
                for d in comp_manifest.directions:
                    comp_lookup[d.shot_id] = d
                    
            prompts = []
            for shot in shot_manifest.shots:
                # We can map shot back to scene via beat_id or micro_beat
                # Find the scene
                matching_scene = None
                if storyboard:
                    # Find scene_id from card
                    for card in storyboard.cards:
                        if card.beat_id == shot.beat_id:
                            matching_scene = scene_lookup.get(card.scene_id)
                            break
                
                mb = micro_beat_lookup.get(shot.shot_id) or micro_beat_lookup.get(shot.shot_id.replace("shot_", ""))
                comp = comp_lookup.get(shot.shot_id)

                # Invoke modular compiler
                positive, negative = compiler.compile(
                    shot=shot,
                    micro_beat=mb,
                    scene=matching_scene,
                    chapter_index=0,  # We don't have chapter-level tracking yet
                    composition=comp
                )

            # Build Legacy AST for DiffusersCompiler
            seed_hash = hashlib.md5(shot.shot_id.encode()).hexdigest()
            seed = int(seed_hash, 16) % (2**32 - 1)
            
            ast_chars = []
            for cm in shot.cast:
                char_ast = CharacterAST(
                    name=cm.character_id,
                    emotion=cm.emotion,
                    pose=cm.pose,
                )
                ast_chars.append(char_ast)

            ast = PromptAST(
                subject=SubjectAST(description=positive),
                characters=ast_chars,
                environment=EnvironmentAST(
                    location=matching_scene.location if matching_scene else "unknown",
                    time_of_day="", weather=""
                ),
                camera=CameraAST(
                    type=shot.camera_type or (mb.purpose if mb else shot.purpose),
                    lens=shot.lens, angle=shot.angle,
                    distance=shot.distance, movement=shot.movement,
                ),
                lighting=LightingAST(style=""),
                composition=CompositionAST(style=""),
                quality=QualityAST(tags=[]),
                negative=NegativeAST(tags=[negative] if negative else []),
            )

            entry = PromptManifestEntry(
                prompt_id=f"prompt_{seed_hash[:8]}",
                scene_id=matching_scene.scene_id if matching_scene else "unknown",
                shot_id=shot.shot_id,
                ast=ast,
                seed=seed,
            )
            prompts.append(entry)

        manifest = PromptManifest(
            prompts=prompts,
            generator="PromptBuilderStage",
            generator_version="4.0.0",
        )

        node = ExecutionNode(artifact=manifest, stage_name="PromptBuilderStage")
        return StageResult(
            artifact=manifest,
            execution_node=node,
            metrics={"prompts_generated": len(prompts)},
            metadata={},
        )
