"""
ShotPlannerStage — deterministic expansion of StoryboardCards into Shots.

This replaces the old LLM-based shot planner.
The LLM decisions (which moments to show) are now made by StoryboardPlannerStage.
This stage simply takes those MicroBeats and maps them 1:1 into physical Shots.

Architecture:
  Beat → StoryboardCard (LLM) → MicroBeats → Shots (Deterministic)
"""
from __future__ import annotations

import logging
from typing import Any

from core.pipeline.stage import CompilerStage, StageResult
from core.domain.assets.execution import ExecutionNode
from core.domain.scene.manifest import ShotManifest, Shot
from core.domain.scene.storyboard import StoryboardManifest
from core.pipeline.context import PipelineContext

logger = logging.getLogger(__name__)


class ShotPlannerStage(CompilerStage):
    """
    Deterministic stage: StoryboardCard.MicroBeat → Shot.
    """

    def __init__(self, llm_provider=None):
        # We don't use the LLM anymore, but keep the signature for backwards compatibility
        # in compiler_api.py instantiation if needed.
        self.llm = llm_provider

    def get_name(self) -> str:
        return "ShotPlannerStage"

    def get_providers(self) -> list:
        return []

    def inputs(self, context: PipelineContext) -> list[Any]:
        results = []
        for node in context.execution_nodes:
            if isinstance(node.artifact, StoryboardManifest):
                results.append(node.artifact)
        return results

    def outputs(self) -> list[str]:
        return ["shot_manifest"]

    def generator_signature(self) -> str:
        return f"{self.get_name()}_deterministic_v2.0"

    def execute(self, context: PipelineContext) -> StageResult:
        storyboard_manifest: StoryboardManifest | None = None
        for node in context.execution_nodes:
            if isinstance(node.artifact, StoryboardManifest):
                storyboard_manifest = node.artifact
                break

        if not storyboard_manifest:
            raise ValueError("ShotPlannerStage: No StoryboardManifest found in context.")

        all_shots = []

        for card in storyboard_manifest.cards:
            scene_id = card.scene_id
            beat_id = card.beat_id

            for mb_idx, mb in enumerate(card.micro_beats):
                # We need a unique ID per shot.
                # Format: shot_{scene_hash}_{beat_id_suffix}_{mb_idx:03d}
                # However, to maintain compatibility with existing hash lookups,
                # we use the micro_beat_id directly if possible, or construct one.
                # Assuming micro_beat_id is already somewhat unique from StoryboardPlanner.
                shot_id = f"shot_{mb.micro_beat_id}" 

                shot = Shot(
                    shot_id=shot_id,
                    beat_id=beat_id,
                    purpose=mb.purpose,
                    emotion=mb.emotion,
                    importance=mb.importance,
                    focus=mb.visual_focus,
                    duration=mb.duration_hint,
                )
                all_shots.append(shot)

        manifest = ShotManifest(shots=all_shots)
        node = ExecutionNode(artifact=manifest, stage_name=self.get_name())

        return StageResult(
            artifact=manifest,
            execution_node=node,
            metrics={"total_shots": len(all_shots)},
            metadata={},
        )
