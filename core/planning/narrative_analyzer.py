"""
NarrativeAnalyzerStage — extracts factual narrative metadata from the StoryBible.

This is the ONLY LLM call in the Direction Layer.
It asks the LLM to read the StoryBible and extract facts about the story:
  • genre, subgenre
  • emotional arc
  • pacing hints
  • recurring motifs and symbols
  • protagonist / antagonist character IDs

The output (NarrativeFacts) is then passed to DirectorPolicyBuilderStage,
which deterministically derives camera + pacing + composition rules from those facts.

Key design principle:
  • LLM produces *facts* (what kind of story is this?)
  • Deterministic engine produces *decisions* (therefore, use push_in for rage emotions)
"""
from __future__ import annotations

import json
import logging
from typing import Any

from core.pipeline.stage import CompilerStage, StageResult
from core.domain.assets.execution import ExecutionNode
from core.domain.story.bible import StoryBible
from core.domain.story.director_manifest import NarrativeFacts, DirectorManifest, DirectorPolicy
from core.pipeline.context import PipelineContext
from core.utils.llm_factory import ensure_llm

logger = logging.getLogger(__name__)


class NarrativeAnalyzerStage(CompilerStage):
    """
    Phase 1 of Direction: LLM extracts narrative facts from StoryBible.
    """

    def __init__(self, llm_provider=None, model_id: str = None, cache_dir: str = None):
        self.llm = llm_provider
        self._model_id = model_id
        self._cache_dir = cache_dir

    def get_name(self) -> str:
        return "NarrativeAnalyzerStage"

    def get_providers(self) -> list:
        return [self.llm] if self.llm else []

    def inputs(self, context: PipelineContext) -> list[Any]:
        for node in context.execution_nodes:
            if isinstance(node.artifact, StoryBible):
                return [node.artifact]
        return []

    def outputs(self) -> list[str]:
        return ["director_manifest"]

    def generator_signature(self) -> str:
        return f"{self.get_name()}_v1.0"

    def execute(self, context: PipelineContext) -> StageResult:
        self.llm = ensure_llm(self.llm, model_id=self._model_id, cache_dir=self._cache_dir)

        bible: StoryBible | None = None
        for node in context.execution_nodes:
            if isinstance(node.artifact, StoryBible):
                bible = node.artifact
                break

        if not bible:
            raise ValueError("NarrativeAnalyzerStage: No StoryBible found in context.")

        facts = self._extract_facts(bible)
        policy = self._build_policy(facts)

        manifest = DirectorManifest(facts=facts, policy=policy)

        node = ExecutionNode(artifact=manifest, stage_name=self.get_name())
        return StageResult(
            artifact=manifest,
            execution_node=node,
            metrics={
                "genre": facts.genre,
                "pacing": facts.pacing_hint,
                "style": facts.target_visual_style,
            },
            metadata={},
        )

    # ─────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────

    def _extract_facts(self, bible: StoryBible) -> NarrativeFacts:
        """Ask the LLM to extract narrative facts. Falls back to sensible defaults."""

        # Build a compact summary of the bible for the LLM
        char_names = list(bible.characters.keys())[:10]
        loc_names = list(bible.locations.keys())[:5]

        # Pull first 500 chars of each character description as context
        char_summaries = []
        for cid in char_names:
            char = bible.characters[cid]
            name = getattr(char, "name", cid)
            role = getattr(char, "role", "")
            char_summaries.append(f"{name} ({role})")

        context_summary = (
            f"Characters: {', '.join(char_summaries)}\n"
            f"Locations: {', '.join(loc_names)}\n"
            f"Themes: {getattr(bible, 'themes', [])}\n"
        )

        schema = {
            "genre": "string (e.g. action-fantasy, romance, thriller, cultivation, isekai)",
            "subgenre": "string (optional, e.g. regressor, villainess, dungeon-hunter)",
            "emotional_arc": "string: one of [rising_tension, catharsis, bittersweet, triumph, tragedy]",
            "pacing_hint": "string: one of [slow_burn, kinetic, varied]",
            "color_temperature": "string: one of [warm, cold, neutral, cold_shifting_to_warm, warm_shifting_to_cold]",
            "recurring_motifs": ["list of 2-5 recurring visual motifs, e.g. sword glow, rain, moon"],
            "important_symbols": ["list of 1-4 symbolic objects e.g. the ring, the scroll"],
            "protagonist_ids": ["list of character IDs that are protagonists"],
            "antagonist_ids": ["list of character IDs that are antagonists"],
            "target_visual_style": "string: one of [korean_manhwa_cinematic, dark_fantasy_cinematic, action_thriller, slice_of_life_webtoon]",
        }

        prompt = (
            "You are a story analyst. Based on the following story Bible summary, "
            "extract narrative facts for use in visual direction.\n\n"
            f"Story Bible Summary:\n{context_summary}\n\n"
            "Output ONLY the requested JSON fields. Do not add commentary."
        )

        try:
            result = self.llm.generate_json(prompt, schema)
            logger.info(f"[NarrativeAnalyzer] Extracted facts: genre={result.get('genre')}, "
                        f"pacing={result.get('pacing_hint')}, style={result.get('target_visual_style')}")

            # Build NarrativeFacts, falling back field-by-field if LLM is missing values
            protagonist_ids = result.get("protagonist_ids", [])
            if isinstance(protagonist_ids, str):
                protagonist_ids = [protagonist_ids]
            antagonist_ids = result.get("antagonist_ids", [])
            if isinstance(antagonist_ids, str):
                antagonist_ids = [antagonist_ids]

            return NarrativeFacts(
                genre=result.get("genre", "fantasy"),
                subgenre=result.get("subgenre", ""),
                emotional_arc=result.get("emotional_arc", "rising_tension"),
                pacing_hint=result.get("pacing_hint", "varied"),
                color_temperature=result.get("color_temperature", "cold_shifting_to_warm"),
                recurring_motifs=result.get("recurring_motifs", []),
                important_symbols=result.get("important_symbols", []),
                protagonist_ids=protagonist_ids,
                antagonist_ids=antagonist_ids,
                target_visual_style=result.get("target_visual_style", "korean_manhwa_cinematic"),
            )

        except Exception as e:
            logger.warning(f"[NarrativeAnalyzer] LLM extraction failed ({e}). Using defaults.")
            # Safe defaults that work well for Korean manhwa
            return NarrativeFacts(
                genre="action-fantasy",
                pacing_hint="varied",
                color_temperature="cold_shifting_to_warm",
                target_visual_style="korean_manhwa_cinematic",
            )

    def _build_policy(self, facts: NarrativeFacts) -> DirectorPolicy:
        """
        Deterministically derive policy rules from narrative facts.
        No LLM is called here — this is pure rule-based logic.
        """
        from core.domain.story.director_manifest import (
            DirectorPolicy, PacingPolicy, CinematographyPolicy, CompositionPolicy
        )

        # Pacing policy from pacing_hint
        if facts.pacing_hint == "kinetic":
            pacing = PacingPolicy(
                action_duration_range=(0.4, 1.2),
                dialogue_duration_range=(1.5, 3.0),
                emotion_duration_range=(0.8, 2.0),
                default_duration=1.5,
                transition_style="hard_cut",
                transition_duration=0.0,
            )
        elif facts.pacing_hint == "slow_burn":
            pacing = PacingPolicy(
                action_duration_range=(1.0, 2.5),
                dialogue_duration_range=(3.0, 6.0),
                emotion_duration_range=(2.0, 4.0),
                default_duration=3.5,
                transition_style="dissolve",
                transition_duration=1.0,
            )
        else:  # varied
            pacing = PacingPolicy(
                action_duration_range=(0.5, 1.5),
                dialogue_duration_range=(2.0, 4.0),
                emotion_duration_range=(1.0, 2.5),
                default_duration=2.5,
                transition_style="xfade",
                transition_duration=0.5,
            )

        # Cinematography: adjust protagonist/antagonist framing
        cinematography = CinematographyPolicy(
            protagonist_frame_side="right",
            antagonist_frame_side="left",
        )

        composition = CompositionPolicy()

        # Style → visual language name
        style_name = facts.target_visual_style or "korean_manhwa_cinematic"

        return DirectorPolicy(
            pacing=pacing,
            cinematography=cinematography,
            composition=composition,
            visual_style_name=style_name,
            color_grade_global=facts.color_temperature,
        )
