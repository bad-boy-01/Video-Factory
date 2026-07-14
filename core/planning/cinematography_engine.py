"""
CinematographyEngineStage (upgraded CameraPlannerStage).

Replaces the 7-keyword lookup table with a full semantic cinematography engine
that reads from DirectorManifest.policy to produce intentional camera decisions.

Key improvements over the old CameraPlannerStage:
  1. Reads DirectorManifest.policy (not hardcoded keyword lists)
  2. Maps shot purpose + emotion → distance + angle + movement
  3. Sets ken_burns_direction on each shot (consumed by FFmpegAssemblyStage)
  4. Applies protagonist/antagonist framing convention from DirectorManifest
  5. Duration is set from DirectorManifest.policy.pacing ranges, not hardcoded

Deterministic: No LLM called. Given the same ShotManifest + DirectorManifest,
always produces the same output.
"""
from __future__ import annotations

import copy
import logging
from typing import Any

from core.pipeline.stage import CompilerStage, StageResult
from core.domain.assets.execution import ExecutionNode
from core.domain.scene.manifest import ShotManifest, Shot
from core.domain.story.director_manifest import DirectorManifest, CinematographyPolicy
from core.pipeline.context import PipelineContext

logger = logging.getLogger(__name__)

# Emotion → lens choice (overwrites purpose-based default for close shots)
EMOTION_LENS = {
    "shock":    "24mm",   # slightly distorted wide — heightens urgency
    "rage":     "35mm",
    "sadness":  "85mm",   # compression → intimacy
    "love":     "85mm",
    "awe":      "24mm",
    "suspense": "50mm",
    "neutral":  "50mm",
}

# Purpose → default lens when emotion doesn't override
PURPOSE_LENS = {
    "establishing":  "24mm",
    "environment":   "24mm",
    "action":        "35mm",
    "dialogue":      "50mm",
    "tension":       "50mm",
    "over_shoulder": "50mm",
    "reaction":      "85mm",
    "emotion_peak":  "85mm",
    "insert":        "100mm macro",
}


class CinematographyEngineStage(CompilerStage):
    """
    Deterministic cinematography rule engine.

    Reads DirectorManifest to apply semantic camera decisions to every shot.
    Produces the same output every time given the same inputs.
    """

    def get_name(self) -> str:
        return "CinematographyEngineStage"

    def get_providers(self) -> list:
        return []

    def inputs(self, context: PipelineContext) -> list[Any]:
        results = []
        for node in context.execution_nodes:
            if isinstance(node.artifact, (ShotManifest, DirectorManifest)):
                results.append(node.artifact)
        return results

    def outputs(self) -> list[str]:
        return ["shot_manifest_with_camera"]

    def generator_signature(self) -> str:
        return f"{self.get_name()}_semantic_v2.0"

    def execute(self, context: PipelineContext) -> StageResult:
        shot_manifest: ShotManifest | None = None
        director: DirectorManifest | None = None

        for node in reversed(context.execution_nodes):
            if isinstance(node.artifact, ShotManifest) and shot_manifest is None:
                shot_manifest = node.artifact
            elif isinstance(node.artifact, DirectorManifest) and director is None:
                director = node.artifact

        if not shot_manifest:
            raise ValueError("CinematographyEngineStage: No ShotManifest found in context.")

        if not director:
            logger.warning("No DirectorManifest found; using default cinematography policy.")
            director = DirectorManifest()

        policy = director.policy.cinematography
        pacing = director.policy.pacing

        enriched = copy.deepcopy(shot_manifest)

        for shot in enriched.shots:
            self._apply_cinematography(shot, policy, pacing, director)

        node = ExecutionNode(artifact=enriched, stage_name=self.get_name())
        return StageResult(
            artifact=enriched,
            execution_node=node,
            metrics={"shots_planned": len(enriched.shots)},
            metadata={},
        )

    # ─────────────────────────────────────────────────────────────────────

    def _apply_cinematography(self, shot: Shot, policy: CinematographyPolicy, pacing, director) -> None:
        """Apply all cinematography rules in-place to a single Shot."""

        purpose = shot.purpose.lower()
        emotion = shot.emotion.lower()
        focus = shot.focus.lower()

        shot.camera_type = "cinematic"

        # ── Distance (from policy table, with focus override) ────────────
        shot.distance = policy.purpose_to_distance.get(purpose, "medium shot")
        if focus == "object":
            shot.distance = "extreme close-up"

        # ── Angle (from policy table, with power-dynamic override) ───────
        shot.angle = policy.purpose_to_angle.get(purpose, "eye-level")

        # Power dynamics: low angle for rage/power, high for vulnerability/grief
        if emotion in ("rage", "power", "intimidating"):
            shot.angle = "low angle"
        elif emotion in ("vulnerable", "weak", "grief", "despair"):
            shot.angle = "high angle"

        # ── Lens ─────────────────────────────────────────────────────────
        shot.lens = EMOTION_LENS.get(emotion, PURPOSE_LENS.get(purpose, "50mm"))

        # ── Movement (Ken Burns direction) ───────────────────────────────
        movement = policy.emotion_to_movement.get(emotion, "push_in")
        shot.movement = movement

        # ── Duration from pacing policy ──────────────────────────────────
        if not shot.duration or shot.duration == 2.0:
            # Only override if still at default; ShotPlanner's duration hints are respected
            if focus == "action" or purpose == "action":
                low, high = pacing.action_duration_range
            elif purpose in ("dialogue", "over_shoulder"):
                low, high = pacing.dialogue_duration_range
            elif purpose in ("reaction", "emotion_peak"):
                low, high = pacing.emotion_duration_range
            else:
                low = high = pacing.default_duration
            shot.duration = (low + high) / 2.0

        # ── Protagonist/antagonist framing hint ─────────────────────────
        # Stored as a style annotation for PromptCompiler to use
        char_ids = [cm.character_id for cm in shot.cast]
        if any(cid in director.facts.protagonist_ids for cid in char_ids):
            shot.style = shot.style or policy.protagonist_frame_side
        elif any(cid in director.facts.antagonist_ids for cid in char_ids):
            shot.style = shot.style or policy.antagonist_frame_side
