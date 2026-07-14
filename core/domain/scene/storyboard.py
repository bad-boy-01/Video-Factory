"""
StoryboardCard — the intermediate representation between a narrative Beat
and a rendered Shot sequence.

Architecture:
  Beat  →  StoryboardCard  →  Shot[]

The StoryboardCard is produced by StoryboardPlannerStage (LLM, cached).
The Shot[] expansion is done deterministically by ShotPlannerStage.

This separation means:
  • Regenerating only the shot sequence (different angles, durations) does
    NOT require re-running the LLM.
  • The LLM's creative decisions (which moments to show) are stable.
  • The physical camera decisions (angle, lens, movement) are deterministic.
"""
from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class MicroBeat(BaseModel):
    """
    A single distinct visual moment within a beat.
    Produced by the LLM inside StoryboardPlannerStage.
    """
    micro_beat_id: str
    description: str
    """What the camera should show — written as a visual description, not prose."""

    visual_focus: str = "character"
    """What the frame foregrounds: 'character' | 'object' | 'environment' | 'reaction' | 'action'"""

    purpose: str = "mid"
    """Shot purpose keyword: 'establishing' | 'reaction' | 'insert' | 'dialogue' |
       'action' | 'emotion_peak' | 'tension' | 'over_shoulder' | 'environment'"""

    emotion: str = "neutral"
    """Dominant emotion of this micro-beat."""

    duration_hint: float = 2.0
    """Suggested display duration in seconds. CinematographyEngine may override."""

    cut_type: str = "soft"
    """How to cut TO this micro-beat: 'hard' | 'soft' | 'smash' | 'dissolve'"""

    importance: str = "medium"
    """Render priority if budget-constrained: 'high' | 'medium' | 'low'"""


class StoryboardCard(BaseModel):
    """
    A director's storyboard card for one narrative beat.

    One Beat → one StoryboardCard → N MicroBeats → N Shots.
    """
    card_id: str
    beat_id: str
    scene_id: str

    purpose: str = ""
    """High-level cinematic purpose of this beat: e.g. 'Reveal monster', 'Emotional confession'"""

    visual_goal: str = ""
    """What emotional/visual effect should be achieved: 'Create tension', 'Establish scale'"""

    estimated_duration: float = 4.0
    """Total estimated duration of all micro-beats combined."""

    micro_beats: List[MicroBeat] = Field(default_factory=list)
    """Ordered list of visual moments to cover this beat."""

    camera_sequence: List[str] = Field(default_factory=list)
    """Simplified summary of visual subjects in order (for human inspection):
       e.g. ['door', 'hallway', 'monster', 'john_face']"""


class StoryboardManifest(BaseModel):
    """The full set of storyboard cards for all beats across all scenes."""
    cards: List[StoryboardCard] = Field(default_factory=list)
    generator: str = "StoryboardPlannerStage"
    schema_version: str = "1.0"

    def get_card_for_beat(self, beat_id: str) -> Optional[StoryboardCard]:
        for card in self.cards:
            if card.beat_id == beat_id:
                return card
        return None
