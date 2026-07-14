"""
PromptCompiler — modular token-contribution architecture.

Replaces the monolithic _build_prompt() function in prompt_builder.py.

Each module contributes its own tokens independently:
  StyleContributor       → style tokens from VisualStyleBible
  CharacterContributor   → appearance/wardrobe/pose/emotion tokens
  CompositionContributor → composition rules from DirectorManifest
  CameraContributor      → framing/angle/lens/movement tokens
  EnvironmentContributor → location/time/weather tokens
  QualityContributor     → masterpiece/quality booster tokens
  NegativeContributor    → negative prompt tokens

The PromptCompiler concatenates them in priority order (CLIP reads left-to-right;
earlier tokens have higher effective weight).

Priority order (highest → lowest):
  1. Style anchor (global manhwa style tokens)
  2. Shot type anchor (establishing / reaction / insert prefix)
  3. Character appearance
  4. Composition rule
  5. Camera framing
  6. Beat description (visual content)
  7. Emotion/mood
  8. Environment (time/weather — can be truncated at 77-token CLIP limit)
  9. Quality boosters

Token budget tracking: warns if assembled prompt exceeds 60 tokens.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from core.domain.story.bible import StoryBible
from core.domain.scene.manifest import Shot, Scene, Beat
from core.domain.scene.storyboard import MicroBeat
from core.domain.style.visual_style_bible import VisualStyleBible
from core.domain.story.director_manifest import DirectorManifest

logger = logging.getLogger(__name__)

# Token budget for CLIP (hard limit: 77 tokens; warn at 60 to leave headroom)
CLIP_WARN_THRESHOLD = 60
CLIP_HARD_LIMIT = 75  # Conservative — leave room for SDXL special tokens

# Shot type anchor tokens (injected first for maximum CLIP weight)
SHOT_TYPE_ANCHORS: Dict[str, str] = {
    "establishing":  "(wide establishing shot:1.3)",
    "reaction":      "(extreme close-up reaction shot:1.3)",
    "insert":        "(macro detail insert shot:1.2)",
    "dialogue":      "(medium shot dialogue:1.1)",
    "action":        "(dynamic action shot:1.2)",
    "emotion_peak":  "(intense close-up emotion:1.3)",
    "tension":       "(tense medium shot:1.1)",
    "over_shoulder": "(over-shoulder shot:1.1)",
    "environment":   "(wide environment shot:1.2)",
}


@dataclass
class TokenContribution:
    """One module's contribution to the final prompt."""
    source: str       # Module name (for debugging)
    tokens: str       # The actual token string
    priority: int     # Lower number = higher priority (closer to start of prompt)
    negative: bool = False  # If True, goes into the negative prompt


class PromptCompiler:
    """
    Assembles the final positive and negative prompt strings for one shot
    from independent token contributions.

    Usage:
        compiler = PromptCompiler(style_bible, director_manifest, story_bible)
        positive, negative = compiler.compile(shot, micro_beat, scene, chapter_index)
    """

    def __init__(
        self,
        style_bible: Optional[VisualStyleBible] = None,
        director: Optional[DirectorManifest] = None,
        bible: Optional[StoryBible] = None,
    ):
        self.style_bible = style_bible or VisualStyleBible()
        self.director = director or DirectorManifest()
        self.bible = bible

    def compile(
        self,
        shot: Shot,
        micro_beat: Optional[MicroBeat],
        scene: Optional[Scene],
        chapter_index: int = 0,
        composition: Optional[Any] = None, # CompositionDirection
    ) -> tuple[str, str]:
        """
        Returns (positive_prompt, negative_prompt) for the given shot.
        """
        contributions: List[TokenContribution] = []

        # ── 1. Style anchor ───────────────────────────────────────────────
        style_tokens = self.style_bible.build_positive_prefix(
            chapter_index=chapter_index,
            scene_id=scene.scene_id if scene else None,
            shot_id=shot.shot_id,
        )
        if style_tokens:
            contributions.append(TokenContribution(
                source="StyleBible", tokens=style_tokens, priority=10
            ))

        # ── 2. Shot type anchor ───────────────────────────────────────────
        purpose = (micro_beat.purpose if micro_beat else shot.purpose or "").lower()
        anchor = SHOT_TYPE_ANCHORS.get(purpose, "")
        if anchor:
            contributions.append(TokenContribution(
                source="ShotTypeAnchor", tokens=anchor, priority=20
            ))

        # ── 3. Character appearance ───────────────────────────────────────
        char_tokens = self._build_character_tokens(shot)
        if char_tokens:
            contributions.append(TokenContribution(
                source="Characters", tokens=char_tokens, priority=30
            ))

        # ── 4. Composition rule from DirectorManifest and CompositionPlanner ──
        composition_tokens = self.director.get_composition_tokens(purpose)
        comp_parts = []
        if composition_tokens:
            comp_parts.append(composition_tokens)
        if composition:
            if hasattr(composition, 'composition_rules') and composition.composition_rules:
                comp_parts.extend(composition.composition_rules)
            if hasattr(composition, 'focal_regions') and composition.focal_regions:
                comp_parts.extend(composition.focal_regions)
        if comp_parts:
            contributions.append(TokenContribution(
                source="Composition", tokens=", ".join(comp_parts), priority=40
            ))

        # ── 5. Camera framing ─────────────────────────────────────────────
        camera_tokens = self._build_camera_tokens(shot)
        if camera_tokens:
            contributions.append(TokenContribution(
                source="Camera", tokens=camera_tokens, priority=50
            ))

        # ── 6. Beat / micro-beat description ──────────────────────────────
        description = ""
        if micro_beat and micro_beat.description:
            description = micro_beat.description
        elif scene:
            description = scene.location
        if description:
            contributions.append(TokenContribution(
                source="BeatDescription", tokens=description, priority=60
            ))

        # ── 7. Emotion / mood ─────────────────────────────────────────────
        emotion = (micro_beat.emotion if micro_beat else shot.emotion or "").lower()
        if emotion and emotion not in ("neutral", ""):
            contributions.append(TokenContribution(
                source="Emotion", tokens=f"{emotion} mood atmosphere", priority=70
            ))

        # ── 8. Environment (time/weather — lowest priority, truncatable) ──
        env_tokens = self._build_environment_tokens(scene)
        if env_tokens:
            contributions.append(TokenContribution(
                source="Environment", tokens=env_tokens, priority=80
            ))

        # ── 9. Quality boosters ───────────────────────────────────────────
        contributions.append(TokenContribution(
            source="Quality",
            tokens="masterpiece, best quality, sharp focus, highly detailed",
            priority=90
        ))

        # ── Build negative ────────────────────────────────────────────────
        negative = self.style_bible.build_negative_prefix(
            chapter_index=chapter_index,
            scene_id=scene.scene_id if scene else None,
            shot_id=shot.shot_id,
        )

        # Sort by priority and concatenate
        contributions.sort(key=lambda c: c.priority)
        positive_parts = [c.tokens for c in contributions if not c.negative and c.tokens]
        positive = ", ".join(positive_parts)

        # Token budget check (approximate: 1 word ≈ 1.3 tokens)
        approx_tokens = len(positive.split()) * 1.3
        if approx_tokens > CLIP_WARN_THRESHOLD:
            logger.warning(
                f"[PromptCompiler] Shot {shot.shot_id}: estimated {approx_tokens:.0f} tokens "
                f"(warn threshold: {CLIP_WARN_THRESHOLD}). "
                "Environment tokens may be truncated by CLIP."
            )
        if approx_tokens > CLIP_HARD_LIMIT:
            # Trim the lowest-priority (environment) contribution
            positive_parts_trimmed = [
                c.tokens for c in contributions
                if not c.negative and c.tokens and c.priority < 80
            ]
            positive = ", ".join(positive_parts_trimmed)
            logger.warning(
                f"[PromptCompiler] Shot {shot.shot_id}: truncated environment tokens "
                f"to stay within CLIP limit."
            )

        return positive, negative

    # ─────────────────────────────────────────────────────────────────────

    def _build_character_tokens(self, shot: Shot) -> str:
        """Build character appearance tokens for all cast members in this shot."""
        if not shot.cast or not self.bible:
            return ""

        parts = []
        for cast_member in shot.cast:
            cid = cast_member.character_id
            profile = self.bible.characters.get(cid)
            if not profile:
                continue
            app = profile.appearance
            char_tokens = []
            if app.hair:      char_tokens.append(app.hair)
            if app.eyes:      char_tokens.append(app.eyes)
            if app.face:      char_tokens.append(app.face)
            if app.age:       char_tokens.append(app.age)
            if app.clothing:  char_tokens.append(app.clothing)
            if cast_member.emotion and cast_member.emotion not in ("neutral", ""):
                char_tokens.append(f"{cast_member.emotion} expression")
            if cast_member.pose and cast_member.pose not in ("standing", ""):
                char_tokens.append(cast_member.pose)
            if char_tokens:
                parts.append(", ".join(char_tokens))

        return "; ".join(parts)

    def _build_camera_tokens(self, shot: Shot) -> str:
        """Build camera framing tokens."""
        parts = []
        if shot.distance:
            parts.append(shot.distance)
        if shot.angle and shot.angle != "eye-level":
            parts.append(shot.angle)
        if shot.lens:
            parts.append(f"{shot.lens} lens")
        return ", ".join(parts)

    def _build_environment_tokens(self, scene: Optional[Scene]) -> str:
        """Build environment tokens (lowest priority — truncated if over budget)."""
        if not scene:
            return ""
        parts = []
        state = scene.state
        if state:
            if state.time:              parts.append(state.time)
            if state.weather:           parts.append(state.weather)
            if state.season:            parts.append(state.season)
            if state.environment_state: parts.append(state.environment_state)
            if state.lighting:          parts.append(state.lighting)
            if state.palette:           parts.append(state.palette)
        elif scene.location:
            parts.append(scene.location)
        return ", ".join(parts)
