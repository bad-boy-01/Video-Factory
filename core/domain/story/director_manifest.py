"""
DirectorManifest — the output of the Narrative Analyzer + Director Policy pipeline.

This is the project-level "creative brief" that every downstream stage reads from.
It is generated once per novel, serialized to workspace/manifests/director_manifest.json,
and never mutated during rendering.

Split into two halves:
  • NarrativeFacts   — LLM-extracted facts (genre, arc, motifs …)
  • DirectorPolicy   — deterministic rules derived from facts (camera movement, pacing …)
"""
from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# LLM-extracted narrative facts (mutable, source of truth from the novel)
# ─────────────────────────────────────────────────────────────────────────────

class NarrativeFacts(BaseModel):
    """Facts extracted from the novel by the Narrative Analyzer LLM call."""

    genre: str = "fantasy"
    """e.g. 'action-fantasy', 'romance', 'thriller', 'slice-of-life'"""

    subgenre: str = ""
    """e.g. 'cultivation', 'isekai', 'regressor', 'villainess'"""

    emotional_arc: str = "rising_tension"
    """Overall story arc: 'rising_tension' | 'catharsis' | 'bittersweet' | 'triumph'"""

    pacing_hint: str = "varied"
    """Global pacing intention: 'slow_burn' | 'kinetic' | 'varied'"""

    color_temperature: str = "cold_shifting_to_warm"
    """'warm' | 'cold' | 'neutral' | 'cold_shifting_to_warm' | 'warm_shifting_to_cold'"""

    recurring_motifs: List[str] = Field(default_factory=list)
    """Visual motifs that appear repeatedly: ['sword glow', 'rain', 'moon shadow']"""

    important_symbols: List[str] = Field(default_factory=list)
    """Objects / symbols that carry narrative weight: ['the ring', 'the scroll']"""

    protagonist_ids: List[str] = Field(default_factory=list)
    """Character IDs identified as protagonists (frame right by convention)"""

    antagonist_ids: List[str] = Field(default_factory=list)
    """Character IDs identified as antagonists (frame left by convention)"""

    scene_pacing_hints: Dict[str, str] = Field(default_factory=dict)
    """Per-scene pacing override: {scene_id: 'kinetic' | 'slow' | 'medium'}"""

    target_visual_style: str = "korean_manhwa_cinematic"
    """High-level aesthetic direction the LLM recommends"""


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic policy rules (derived from NarrativeFacts, never by LLM)
# ─────────────────────────────────────────────────────────────────────────────

class PacingPolicy(BaseModel):
    """Cut timing rules derived from pacing_hint."""
    action_duration_range: tuple[float, float] = (0.5, 1.5)
    dialogue_duration_range: tuple[float, float] = (2.0, 4.0)
    emotion_duration_range: tuple[float, float] = (1.0, 2.5)
    default_duration: float = 2.5
    transition_style: str = "xfade"       # 'xfade' | 'hard_cut' | 'dissolve'
    transition_duration: float = 0.5


class CinematographyPolicy(BaseModel):
    """Camera movement rules derived from genre + pacing."""

    # Emotion → Ken Burns direction
    emotion_to_movement: Dict[str, str] = Field(default_factory=lambda: {
        "shock":    "static",
        "surprise": "whip_pan",
        "sadness":  "pull_back",
        "grief":    "pull_back",
        "rage":     "push_in",
        "power":    "push_in",
        "wonder":   "crane_up",
        "awe":      "crane_up",
        "suspense": "static",
        "love":     "dolly_in",
        "warmth":   "dolly_in",
        "action":   "pan_right",
        "neutral":  "push_in",
    })

    # Shot purpose → camera distance
    purpose_to_distance: Dict[str, str] = Field(default_factory=lambda: {
        "establishing":  "wide shot",
        "reaction":      "close-up",
        "insert":        "extreme close-up",
        "dialogue":      "medium shot",
        "action":        "medium shot",
        "emotion_peak":  "extreme close-up",
        "tension":       "medium shot",
        "over_shoulder": "medium shot",
        "environment":   "wide shot",
    })

    # Shot purpose → camera angle
    purpose_to_angle: Dict[str, str] = Field(default_factory=lambda: {
        "establishing":  "high angle",
        "reaction":      "eye-level",
        "insert":        "eye-level",
        "dialogue":      "eye-level",
        "action":        "low angle",
        "emotion_peak":  "eye-level",
        "tension":       "eye-level",
        "environment":   "high angle",
    })

    protagonist_frame_side: str = "right"
    antagonist_frame_side: str = "left"


class CompositionPolicy(BaseModel):
    """Composition rule tokens per shot purpose."""

    # Maps shot purpose → composition prompt tokens to inject
    purpose_to_composition_tokens: Dict[str, str] = Field(default_factory=lambda: {
        "establishing": (
            "rule of thirds, subject on left vertical third, "
            "expansive negative space on right, foreground depth element, "
            "atmospheric haze in background, leading lines converging on subject"
        ),
        "reaction": (
            "face centered in frame, extreme shallow depth of field, "
            "background bokeh, eyes at upper third line, "
            "tight framing with slight headroom"
        ),
        "insert": (
            "object fills 80% of frame, abstract blurred background, "
            "rim lighting on subject edges, macro detail focus"
        ),
        "dialogue": (
            "over-shoulder framing, speaking character occupies 60% of frame, "
            "listener in soft focus foreground left, depth layering"
        ),
        "action": (
            "dynamic diagonal composition, motion lines suggest direction, "
            "low horizon line, subject dominates upper two-thirds"
        ),
        "emotion_peak": (
            "centered symmetry, high contrast vignette, "
            "subject fills center third, dramatic negative space above"
        ),
        "environment": (
            "panoramic framing, multiple depth planes (fore/mid/back), "
            "sky occupies upper third, ground texture in foreground"
        ),
        "tension": (
            "claustrophobic framing, subject slightly off-center, "
            "hard shadow cutting across frame, compressed perspective"
        ),
    })


class DirectorPolicy(BaseModel):
    """
    Fully deterministic policy derived from NarrativeFacts.
    No LLM is ever consulted to produce these values.
    """
    pacing: PacingPolicy = Field(default_factory=PacingPolicy)
    cinematography: CinematographyPolicy = Field(default_factory=CinematographyPolicy)
    composition: CompositionPolicy = Field(default_factory=CompositionPolicy)

    visual_style_name: str = "korean_manhwa_cinematic"
    """Key into VisualStyleBible.styles"""

    color_grade_global: str = "cold_shifting_to_warm"
    """Applied as a style token suffix on every prompt"""


# ─────────────────────────────────────────────────────────────────────────────
# Combined manifest
# ─────────────────────────────────────────────────────────────────────────────

class DirectorManifest(BaseModel):
    """
    The project's creative brief.

    Generated by: NarrativeAnalyzerStage (LLM) → DirectorPolicyBuilderStage (deterministic)
    Consumed by: StoryboardPlannerStage, CinematographyEngineStage,
                 CompositionPlannerStage, PromptCompilerStage
    """
    facts: NarrativeFacts = Field(default_factory=NarrativeFacts)
    policy: DirectorPolicy = Field(default_factory=DirectorPolicy)

    generator: str = "NarrativeAnalyzerStage+DirectorPolicyBuilderStage"
    schema_version: str = "1.0"

    def get_movement_for_emotion(self, emotion: str) -> str:
        """Convenience accessor for cinematography policy."""
        return self.policy.cinematography.emotion_to_movement.get(
            emotion.lower(), "push_in"
        )

    def get_distance_for_purpose(self, purpose: str) -> str:
        return self.policy.cinematography.purpose_to_distance.get(
            purpose.lower(), "medium shot"
        )

    def get_angle_for_purpose(self, purpose: str) -> str:
        return self.policy.cinematography.purpose_to_angle.get(
            purpose.lower(), "eye-level"
        )

    def get_composition_tokens(self, purpose: str) -> str:
        return self.policy.composition.purpose_to_composition_tokens.get(
            purpose.lower(), "cinematic framing, rule of thirds"
        )
