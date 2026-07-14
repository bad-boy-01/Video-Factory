"""
StoryboardPlannerStage — expands narrative Beats into StoryboardCards.

Architecture position:
  SceneSplitterStage → StoryboardPlannerStage → ShotPlannerStage

For each Beat, the LLM produces one StoryboardCard containing:
  • A list of MicroBeats (distinct visual moments)
  • A camera_sequence summary
  • An estimated duration

The Shot Planner then deterministically expands each MicroBeat into
a Shot with physical camera parameters from the DirectorManifest policy.

The LLM call is cached (one per beat), so re-runs are instant.
Regenerating only the shot sequence (different angles/durations) does NOT
require re-running this stage.
"""
from __future__ import annotations

import logging
import hashlib
from typing import Any, List

from core.pipeline.stage import CompilerStage, StageResult
from core.domain.assets.execution import ExecutionNode
from core.domain.scene.manifest import SceneManifest, Beat
from core.domain.scene.storyboard import (
    StoryboardCard, StoryboardManifest, MicroBeat
)
from core.domain.story.director_manifest import DirectorManifest
from core.pipeline.context import PipelineContext
from core.utils.llm_factory import ensure_llm
from core.utils.scene_hash import compute_scene_hash

logger = logging.getLogger(__name__)

# Micro-beat count per scene style
MICRO_BEAT_COUNTS = {
    "action":    (5, 8),
    "narrative": (3, 5),
    "montage":   (2, 4),
    "default":   (3, 6),
}


class StoryboardPlannerStage(CompilerStage):
    """
    LLM stage: Beat → StoryboardCard (with MicroBeats).

    One LLM call per beat. Heavily cached.
    """

    def __init__(self, llm_provider=None, model_id: str = None, cache_dir: str = None):
        self.llm = llm_provider
        self._model_id = model_id
        self._cache_dir = cache_dir

    def get_name(self) -> str:
        return "StoryboardPlannerStage"

    def get_providers(self) -> list:
        return [self.llm] if self.llm else []

    def inputs(self, context: PipelineContext) -> list[Any]:
        results = []
        for node in context.execution_nodes:
            if isinstance(node.artifact, (SceneManifest, DirectorManifest)):
                results.append(node.artifact)
        return results

    def outputs(self) -> list[str]:
        return ["storyboard_manifest"]

    def generator_signature(self) -> str:
        return f"{self.get_name()}_v1.0"

    def execute(self, context: PipelineContext) -> StageResult:
        self.llm = ensure_llm(self.llm, model_id=self._model_id, cache_dir=self._cache_dir)

        scene_manifest: SceneManifest | None = None
        director: DirectorManifest | None = None

        for node in context.execution_nodes:
            if isinstance(node.artifact, SceneManifest):
                scene_manifest = node.artifact
            elif isinstance(node.artifact, DirectorManifest):
                director = node.artifact

        if not scene_manifest:
            raise ValueError("StoryboardPlannerStage: No SceneManifest found in context.")

        if not director:
            logger.warning("No DirectorManifest found; using default pacing.")
            from core.domain.story.director_manifest import DirectorManifest as DM
            director = DM()

        cards: list[StoryboardCard] = []
        total_micro_beats = 0

        for scene in scene_manifest.scenes:
            scene_hash = compute_scene_hash(scene.scene_id)

            for beat in scene.beats:
                card = self._plan_beat(scene, beat, scene_hash, director)
                cards.append(card)
                total_micro_beats += len(card.micro_beats)

        manifest = StoryboardManifest(cards=cards)

        node = ExecutionNode(artifact=manifest, stage_name=self.get_name())
        return StageResult(
            artifact=manifest,
            execution_node=node,
            metrics={
                "cards": len(cards),
                "micro_beats": total_micro_beats,
                "avg_micro_beats_per_beat": round(total_micro_beats / max(1, len(cards)), 1),
            },
            metadata={},
        )

    # ─────────────────────────────────────────────────────────────────────

    def _plan_beat(
        self,
        scene,
        beat: Beat,
        scene_hash: str,
        director: DirectorManifest,
    ) -> StoryboardCard:

        pacing = director.facts.pacing_hint
        min_mb, max_mb = MICRO_BEAT_COUNTS.get(pacing, MICRO_BEAT_COUNTS["default"])

        schema = {
            "purpose": "string — the cinematic purpose of this beat (e.g. 'Reveal monster', 'Emotional confession')",
            "visual_goal": "string — what visual/emotional effect should be achieved (e.g. 'Create tension', 'Establish scale')",
            "estimated_duration": "number — total seconds for this beat",
            "camera_sequence": ["list of 3-8 strings: visual subjects in order, e.g. 'door', 'hallway', 'monster', 'john_face'"],
            "micro_beats": [
                {
                    "micro_beat_id": "string",
                    "description": "string — what the camera shows, written as a visual description",
                    "visual_focus": "string: one of [character, object, environment, reaction, action]",
                    "purpose": "string: one of [establishing, reaction, insert, dialogue, action, emotion_peak, tension, over_shoulder, environment]",
                    "emotion": "string — dominant emotion",
                    "duration_hint": "number — suggested seconds",
                    "cut_type": "string: one of [hard, soft, smash, dissolve]",
                    "importance": "string: one of [high, medium, low]"
                }
            ]
        }

        style_context = director.facts.target_visual_style

        prompt = f"""You are a Korean manhwa story director ({style_context} style).
Expand the following narrative beat into {min_mb}–{max_mb} distinct visual micro-beats for a storyboard.

RULES:
- Every micro-beat must show a DIFFERENT visual subject from the previous
- Include at least ONE reaction shot (character's face)  
- Include at least ONE environmental anchor (location detail)
- Action sequences: prefer 'smash' and 'hard' cuts, short durations (0.5–1.2s)
- Emotional moments: prefer 'soft' or 'dissolve' cuts, longer durations (1.5–3.0s)
- Write descriptions as VISUAL directions, not prose: "Monster claw swiping toward camera" not "The monster attacked"

Scene Location: {scene.location}
Scene Characters: {', '.join(scene.characters)}

Beat to expand:
  Beat ID: {beat.beat_id}
  Description: {beat.description}
  Emotion: {beat.emotion}
"""

        card_id = f"card_{scene_hash}_{beat.beat_id}"

        try:
            result = self.llm.generate_json(prompt, schema)

            micro_beats_raw = result.get("micro_beats", [])
            micro_beats = []
            for i, mb in enumerate(micro_beats_raw):
                micro_beats.append(MicroBeat(
                    micro_beat_id=mb.get("micro_beat_id", f"mb_{i:03d}"),
                    description=mb.get("description", beat.description),
                    visual_focus=mb.get("visual_focus", "character"),
                    purpose=mb.get("purpose", "mid"),
                    emotion=mb.get("emotion", beat.emotion),
                    duration_hint=float(mb.get("duration_hint", 2.0)),
                    cut_type=mb.get("cut_type", "soft"),
                    importance=mb.get("importance", "medium"),
                ))

            if not micro_beats:
                micro_beats = self._fallback_micro_beats(beat)

            return StoryboardCard(
                card_id=card_id,
                beat_id=beat.beat_id,
                scene_id=scene.scene_id,
                purpose=result.get("purpose", "cover beat"),
                visual_goal=result.get("visual_goal", "convey emotion"),
                estimated_duration=float(result.get("estimated_duration", 4.0)),
                camera_sequence=result.get("camera_sequence", []),
                micro_beats=micro_beats,
            )

        except Exception as e:
            logger.warning(
                f"[StoryboardPlanner] LLM failed for beat {beat.beat_id} ({e}). "
                "Using fallback storyboard."
            )
            return StoryboardCard(
                card_id=card_id,
                beat_id=beat.beat_id,
                scene_id=scene.scene_id,
                purpose="cover beat",
                visual_goal="convey emotion",
                estimated_duration=4.0,
                micro_beats=self._fallback_micro_beats(beat),
            )

    def _fallback_micro_beats(self, beat: Beat) -> List[MicroBeat]:
        """Generate minimal safe micro-beats when the LLM fails."""
        return [
            MicroBeat(
                micro_beat_id="mb_000",
                description=f"Establishing: {beat.description}",
                visual_focus="environment",
                purpose="establishing",
                emotion=beat.emotion,
                duration_hint=2.5,
                cut_type="soft",
                importance="high",
            ),
            MicroBeat(
                micro_beat_id="mb_001",
                description=f"Character reaction to: {beat.description}",
                visual_focus="reaction",
                purpose="reaction",
                emotion=beat.emotion,
                duration_hint=1.5,
                cut_type="soft",
                importance="high",
            ),
        ]
