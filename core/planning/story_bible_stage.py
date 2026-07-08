from core.pipeline.stage import CompilerStage, StageResult
from core.domain.assets.execution import ExecutionNode
from core.domain.story.bible import StoryBible, CharacterVisualProfile, Location, Appearance
from core.pipeline.context import PipelineContext
from core.planning.chunker import ChunkerManifest
from typing import Any, Dict
import logging

logger = logging.getLogger(__name__)


def _non_empty_field_count(obj_dict: dict) -> int:
    """Counts populated fields, used to decide which of two extractions of the
    same character/location to keep when they're seen again in a later chunk."""
    count = 0
    for v in obj_dict.values():
        if isinstance(v, list):
            count += len([x for x in v if x])
        elif v and str(v).strip().lower() not in {"unknown", "not specified", "none"}:
            count += 1
    return count


class StoryBibleGeneratorStage(CompilerStage):
    def __init__(self, llm_provider, rag_index=None, history_path=None):
        self.llm = llm_provider
        # Optional NovelRAGIndex - when provided, each chunk's extraction
        # prompt is enriched with the most relevant earlier chunks, so a
        # character/location re-described 30 chunks later doesn't silently
        # drift from how it was first established. Retrieval is best-effort:
        # if rag_index is None or unavailable, this stage behaves exactly as
        # it did before.
        self.rag_index = rag_index
        # Optional path to persist every character/location version seen
        # (not just the merged result) - lets a person inspect and roll back
        # if a later "richer" extraction turns out to be a hallucination.
        self.history_path = history_path

    def get_name(self) -> str:
        return "StoryBibleGeneratorStage"

    def get_providers(self) -> list:
        return [self.llm] if self.llm else []

    def inputs(self, context: PipelineContext) -> list[Any]:
        return [context.project_manifest]

    def outputs(self) -> list[str]:
        return ["story_bible"]

    def generator_signature(self) -> str:
        return f"{self.get_name()}_{type(self.llm).__name__ if self.llm else 'default'}_v3.0"

    def execute(self, context: PipelineContext) -> StageResult:
        schema = {
            "characters": [
                {
                    "id": "string",
                    "name": "string",
                    "appearance": {
                        "hair": "string",
                        "eyes": "string",
                        "face": "string",
                        "age": "string",
                        "body": "string",
                        "clothing": "string",
                        "color_palette": ["string"],
                        "signature": ["string"]
                    }
                }
            ],
            "locations": [
                {
                    "id": "string",
                    "name": "string",
                    "appearance": "string",
                    "architecture": "string",
                    "weather_defaults": "string",
                    "time_defaults": "string",
                    "lighting_presets": "string"
                }
            ]
        }

        chunker_manifest = None
        for node in reversed(context.execution_nodes):
            if isinstance(node.artifact, ChunkerManifest):
                chunker_manifest = node.artifact
                break

        if not chunker_manifest:
            raise ValueError("StoryBibleGenerator: Missing ChunkerManifest. Did ChunkerStage run?")

        chars: Dict[str, CharacterVisualProfile] = {}
        char_raw: Dict[str, dict] = {}  # tracks the raw dict behind each stored profile, for dedup scoring
        locs: Dict[str, Location] = {}
        loc_raw: Dict[str, dict] = {}

        history = None
        if self.history_path:
            from core.memory.character_history import CharacterHistoryStore
            history = CharacterHistoryStore(self.history_path)

        for chunk in chunker_manifest.chunks:
            retrieved_context = ""
            if self.rag_index is not None:
                results = self.rag_index.retrieve(
                    chunk.text, before_chunk_id=chunk.chunk_id, top_k=3
                )
                retrieved_context = self.rag_index.format_context(results)

            context_block = f"\n\n{retrieved_context}\n" if retrieved_context else ""

            prompt = f"""
You are an expert cinematic production designer. Extract the following from this
chunk of the story:
1. Detailed Character Visual Profiles (physical traits, signature accessories, clothing).
2. Locations (architecture, weather, lighting defaults).

Only extract characters/locations that appear or are meaningfully referenced in
THIS chunk - do not invent ones from the earlier context below, which is only
provided to help you keep descriptions consistent with what was already
established.
{context_block}
Chunk (chapter {chunk.chapter}):
{chunk.text}
"""
            try:
                result_dict = self.llm.generate_json(prompt, schema)
            except Exception as e:
                logger.warning(f"StoryBible extraction failed for a chunk: {e}")
                result_dict = {"characters": [], "locations": []}

            for c in result_dict.get("characters", []):
                try:
                    char_id = c.get("id", c.get("name", "")).strip().lower().replace(" ", "_")
                    if not char_id:
                        continue
                    app_data = c.get("appearance", {})
                    name = c.get("name", "Unknown")

                    # A character mentioned again in a later chunk without a
                    # full re-description shouldn't overwrite a richer profile
                    # already captured - keep whichever extraction has more
                    # populated fields. Ties keep the existing (earlier) one.
                    is_richer = (
                        char_id not in char_raw
                        or _non_empty_field_count(app_data) > _non_empty_field_count(char_raw[char_id])
                    )

                    if history is not None:
                        history.record(
                            "characters", char_id, chunk.chunk_id,
                            {"name": name, "appearance": app_data}, is_active=is_richer,
                        )

                    if not is_richer:
                        continue

                    chars[char_id] = CharacterVisualProfile(
                        id=char_id,
                        name=name,
                        appearance=Appearance(**app_data)
                    )
                    char_raw[char_id] = app_data
                except Exception:
                    pass

            for l in result_dict.get("locations", []):
                try:
                    loc_id = l.get("id", l.get("name", "")).strip().lower().replace(" ", "_")
                    if not loc_id:
                        continue

                    is_richer = (
                        loc_id not in loc_raw
                        or _non_empty_field_count(l) > _non_empty_field_count(loc_raw[loc_id])
                    )

                    if history is not None:
                        history.record("locations", loc_id, chunk.chunk_id, l, is_active=is_richer)

                    if not is_richer:
                        continue

                    locs[loc_id] = Location(**l)
                    loc_raw[loc_id] = l
                except Exception:
                    pass

            if self.rag_index is not None:
                self.rag_index.add_chunk(chunk.chunk_id, chunk.text)

        if history is not None:
            history.save()

        bible = StoryBible(
            characters=chars,
            locations=locs
        )

        node = ExecutionNode(artifact=bible, stage_name="StoryBibleGeneratorStage")

        return StageResult(
            artifact=bible,
            execution_node=node,
            metrics={"characters": len(chars), "locations": len(locs), "chunks_processed": len(chunker_manifest.chunks)},
            metadata={}
        )
