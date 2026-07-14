"""
SceneGraphBuilderStage — builds a structured SceneGraph for every scene.

Parses the scene description and beats to identify all characters, objects, and
environment elements present, and the spatial/semantic relationships between them.

This creates the `SceneGraphManifest` which is used by later stages (like CompositionPlanner)
to generate layout/positional hints for the diffusion model.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from core.pipeline.stage import CompilerStage, StageResult
from core.domain.assets.execution import ExecutionNode
from core.domain.scene.manifest import SceneManifest
from core.domain.story.bible import StoryBible
from core.domain.scene.graph import (
    SceneGraphManifest, SceneGraph, SceneEntity, EntityState, Relationship, EntityType
)
from core.pipeline.context import PipelineContext
from core.utils.llm_factory import ensure_llm

logger = logging.getLogger(__name__)


class SceneGraphBuilderStage(CompilerStage):
    """
    LLM Stage: Parses scenes to build relationship graphs.
    """

    def __init__(self, llm_provider=None, model_id: str = None, cache_dir: str = None):
        self.llm = llm_provider
        self._model_id = model_id
        self._cache_dir = cache_dir

    def get_name(self) -> str:
        return "SceneGraphBuilderStage"

    def get_providers(self) -> list:
        return [self.llm] if self.llm else []

    def inputs(self, context: PipelineContext) -> list[Any]:
        results = []
        for node in context.execution_nodes:
            if isinstance(node.artifact, (SceneManifest, StoryBible)):
                results.append(node.artifact)
        return results

    def outputs(self) -> list[str]:
        return ["scene_graph_manifest"]

    def generator_signature(self) -> str:
        return f"{self.get_name()}_v1.0"

    def execute(self, context: PipelineContext) -> StageResult:
        self.llm = ensure_llm(self.llm, model_id=self._model_id, cache_dir=self._cache_dir)

        scene_manifest: SceneManifest | None = None
        bible: StoryBible | None = None

        for node in context.execution_nodes:
            if isinstance(node.artifact, SceneManifest):
                scene_manifest = node.artifact
            elif isinstance(node.artifact, StoryBible):
                bible = node.artifact

        if not scene_manifest:
            raise ValueError("SceneGraphBuilderStage: No SceneManifest found in context.")
        if not bible:
            raise ValueError("SceneGraphBuilderStage: No StoryBible found in context.")

        graphs: Dict[str, SceneGraph] = {}

        for scene in scene_manifest.scenes:
            graph = self._build_graph(scene, bible)
            graphs[scene.scene_id] = graph

        manifest = SceneGraphManifest(graphs=graphs)
        node = ExecutionNode(artifact=manifest, stage_name=self.get_name())

        return StageResult(
            artifact=manifest,
            execution_node=node,
            metrics={"total_graphs": len(graphs)},
            metadata={},
        )

    def _build_graph(self, scene, bible: StoryBible) -> SceneGraph:
        """Use LLM to extract entities and relationships for one scene."""
        schema = {
            "entities": [
                {
                    "entity_id": "string (e.g. char_john, obj_sword, env_cabin)",
                    "entity_type": "string: one of [character, object, environment, location, animal, vehicle]",
                    "display_name": "string",
                    "position": "string (e.g. center, background-right)",
                    "physical_state": "string (e.g. standing, glowing)",
                    "character_profile_id": "string (only for characters, matching StoryBible ID, or empty)"
                }
            ],
            "relationships": [
                {
                    "from_entity_id": "string",
                    "relation": "string (e.g. holds, behind, faces, lights)",
                    "to_entity_id": "string"
                }
            ],
            "ambient_conditions": {
                "weather": "string",
                "time": "string",
                "lighting": "string"
            }
        }

        beats_text = "\n".join([f"- {b.description}" for b in scene.beats])

        prompt = f"""You are a scene layout planner. Read the following scene and extract all present entities (characters, important objects, environment elements) and their spatial/semantic relationships.

Location: {scene.location}
Characters present: {', '.join(scene.characters)}
Action/Beats:
{beats_text}

Extract the entities and relationships into the requested JSON schema.
Ensure from_entity_id and to_entity_id in relationships exactly match the entity_id of extracted entities.
"""
        
        try:
            result = self.llm.generate_json(prompt, schema)
            
            graph = SceneGraph(scene_id=scene.scene_id, location_id=scene.location)
            
            for ent_data in result.get("entities", []):
                ent_type = ent_data.get("entity_type", "object")
                try:
                    et = EntityType(ent_type)
                except:
                    et = EntityType.OBJECT
                    
                entity = SceneEntity(
                    entity_id=ent_data.get("entity_id", "unknown"),
                    entity_type=et,
                    display_name=ent_data.get("display_name", ""),
                    character_profile_id=ent_data.get("character_profile_id") or None,
                    state=EntityState(
                        position=ent_data.get("position", ""),
                        physical_state=ent_data.get("physical_state", "")
                    )
                )
                graph.entities[entity.entity_id] = entity
                
            for rel_data in result.get("relationships", []):
                from_id = rel_data.get("from_entity_id", "")
                to_id = rel_data.get("to_entity_id", "")
                if from_id in graph.entities and to_id in graph.entities:
                    rel = Relationship(
                        from_entity_id=from_id,
                        relation=rel_data.get("relation", "near"),
                        to_entity_id=to_id
                    )
                    graph.relationships.append(rel)
                    
            graph.ambient_conditions = result.get("ambient_conditions", {})
            return graph
            
        except Exception as e:
            logger.warning(f"[SceneGraphBuilder] LLM failed for scene {scene.scene_id}: {e}")
            return SceneGraph(scene_id=scene.scene_id)
