"""
CompositionPlannerStage — Maps SceneGraph entities and Relationships into compositional directions.
"""
from __future__ import annotations
import logging
import json
from typing import Any, List

from core.pipeline.stage import CompilerStage, StageResult
from core.domain.assets.execution import ExecutionNode
from core.domain.scene.graph import SceneGraphManifest
from core.domain.scene.storyboard import StoryboardManifest
from core.domain.scene.manifest import ShotManifest
from core.pipeline.context import PipelineContext
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class CompositionDirection(BaseModel):
    shot_id: str
    composition_rules: List[str] = Field(default_factory=list)
    focal_regions: List[str] = Field(default_factory=list)

class CompositionManifest(BaseModel):
    directions: List[CompositionDirection] = Field(default_factory=list)

class CompositionPlannerStage(CompilerStage):
    def get_name(self) -> str:
        return "CompositionPlannerStage"

    def get_providers(self) -> list:
        return []

    def inputs(self, context: PipelineContext) -> list[Any]:
        inputs = []
        for node in context.execution_nodes:
            if isinstance(node.artifact, (SceneGraphManifest, StoryboardManifest, ShotManifest)):
                inputs.append(node.artifact)
        return inputs

    def outputs(self) -> list[str]:
        return ["composition_manifest"]

    def generator_signature(self) -> str:
        return "CompositionPlannerStage_v1.0"

    def execute(self, context: PipelineContext) -> StageResult:
        scene_graph: SceneGraphManifest = None
        shot_manifest: ShotManifest = None
        
        for node in reversed(context.execution_nodes):
            if scene_graph is None and isinstance(node.artifact, SceneGraphManifest):
                scene_graph = node.artifact
            elif shot_manifest is None and isinstance(node.artifact, ShotManifest):
                shot_manifest = node.artifact
                
        if not scene_graph or not shot_manifest:
            logger.warning("CompositionPlannerStage requires SceneGraphManifest and ShotManifest. Skipping.")
            manifest = CompositionManifest()
            return StageResult(
                artifact=manifest,
                execution_node=ExecutionNode(artifact=manifest, stage_name=self.get_name()),
                metrics={}, metadata={}
            )
            
        directions = []
        
        # Build lookup for scene graphs
        sg_lookup = {sg.scene_id: sg for sg in scene_graph.graphs}
        
        for shot in shot_manifest.shots:
            # We don't have direct scene_id in shot in the legacy manifest, 
            # but SceneGraph has scene_ids. The shot typically comes from a scene.
            # We can use the relationships from any scene graph if we can match entities.
            
            rules = []
            focal = []
            
            shot_chars = [c.character_id for c in shot.cast]
            
            if len(shot_chars) == 1:
                rules.append("Rule of thirds, centered subject")
                focal.append("subject in focus, bokeh background")
            elif len(shot_chars) == 2:
                rules.append("Over-the-shoulder or balanced two-shot")
                # Look for relationships between these two characters in scene graphs
                rel_found = False
                for sg in scene_graph.graphs:
                    for rel in sg.relationships:
                        if (rel.source in shot_chars and rel.target in shot_chars):
                            if "combat" in rel.relation_type.lower() or "fight" in rel.relation_type.lower():
                                rules.append("Dynamic Dutch angle, high contrast framing")
                                focal.append("action-focused, intense eye contact")
                                rel_found = True
                            elif "romance" in rel.relation_type.lower() or "intimate" in rel.relation_type.lower():
                                rules.append("Close proximity, shallow depth of field")
                                rel_found = True
                
                if not rel_found:
                    rules.append("balanced split composition")
            else:
                rules.append("Wide establishing group shot")
                
            directions.append(CompositionDirection(
                shot_id=shot.shot_id,
                composition_rules=rules,
                focal_regions=focal
            ))
            
        manifest = CompositionManifest(directions=directions)
        
        return StageResult(
            artifact=manifest,
            execution_node=ExecutionNode(artifact=manifest, stage_name=self.get_name()),
            metrics={"directions_generated": len(directions)},
            metadata={}
        )
