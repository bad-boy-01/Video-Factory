from core.pipeline.stage import CompilerStage, StageResult
from core.domain.assets.execution import ExecutionNode
from core.domain.prompt.visual_scene import VisualScene
from core.domain.graph.scene_graph import SceneGraph, GraphNode, GraphEdge
from core.pipeline.context import PipelineContext
from typing import Any
import logging

logger = logging.getLogger(__name__)

class SceneGraphBuilderStage(CompilerStage):
    def __init__(self, llm_provider=None):
        self.llm = llm_provider

    def get_name(self) -> str:
        return "SceneGraphBuilderStage"

    def get_providers(self) -> list:
        return [self.llm] if self.llm else []
        
    def inputs(self, context: PipelineContext) -> list[Any]:
        # We need a list of VisualScenes. They are likely in a manifest.
        # Actually in prompt_builder we just generated them. Let's assume there is a PromptManifest 
        # (or VisualSceneManifest) containing VisualScenes.
        return [] # We'll extract from context
        
    def outputs(self) -> list[str]:
        return ["scene_graph"]
        
    def generator_signature(self) -> str:
        return f"{self.get_name()}_v1.0"

    def execute(self, context: PipelineContext) -> StageResult:
        if not self.llm:
            from plugins.local_llm import LocalLLMProvider
            self.llm = LocalLLMProvider()
            
        from core.domain.prompt.ast import PromptManifest
        
        prompt_manifest = None
        for node in context.execution_nodes:
            if isinstance(node.artifact, PromptManifest):
                prompt_manifest = node.artifact
                
        if not prompt_manifest:
            raise ValueError("SceneGraphBuilder requires PromptManifest containing VisualScenes.")
            
        scene_graphs = []
        for entry in prompt_manifest.prompts:
            vs = entry.visual_scene
            if not vs:
                continue
                
            nodes = []
            edges = []
            
            # Add Characters
            for c in vs.characters:
                nodes.append(GraphNode(
                    id=c.name,
                    type="Character",
                    properties={
                        "appearance": c.appearance,
                        "wardrobe": c.wardrobe,
                        "pose": c.pose,
                        "emotion": c.emotion
                    }
                ))
            
            # Add Environment
            env_id = "Environment_1"
            nodes.append(GraphNode(
                id=env_id,
                type="Environment",
                properties={
                    "location_desc": vs.environment.location_desc,
                    "time": vs.environment.time,
                    "weather": vs.environment.weather
                }
            ))
            
            # Add Camera
            cam_id = "Camera_1"
            nodes.append(GraphNode(
                id=cam_id,
                type="Camera",
                properties={
                    "type": vs.camera.type,
                    "lens": vs.camera.lens,
                    "distance": vs.camera.distance
                }
            ))
            
            # Add Light
            light_id = "Light_1"
            nodes.append(GraphNode(
                id=light_id,
                type="Light",
                properties={
                    "lighting_style": vs.style.lighting_style
                }
            ))
            
            # Hybrid extraction: LLM analyzes action and we deterministically create edges
            schema = {
                "edges": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source_id": {"type": "string"},
                            "target_id": {"type": "string"},
                            "relationship": {"type": "string"}
                        },
                        "required": ["source_id", "target_id", "relationship"]
                    }
                }
            }
            
            char_interactions = " ".join([f"{c.name} action: {c.interaction}" for c in vs.characters])
            prompt = f"""
Given these character actions: {char_interactions}
And subject: {vs.subject}

Extract explicit visual relationships as graph edges.
Source/Target must be character names, 'Environment_1', or 'Camera_1'.
Allowed relationships: 'holding', 'looking_at', 'standing_on', 'inside', 'wearing'.
"""
            try:
                result = self.llm.generate_json(prompt, schema)
                for edge_data in result.get("edges", []):
                    edges.append(GraphEdge(
                        source_id=edge_data["source_id"],
                        target_id=edge_data["target_id"],
                        relationship=edge_data["relationship"]
                    ))
            except Exception as e:
                logger.warning(f"Failed to extract edges for {entry.shot_id}")
                
            sg = SceneGraph(nodes=nodes, edges=edges)
            scene_graphs.append((entry.shot_id, sg))
            
        # In a real pipeline, we'd return a SceneGraphManifest.
        # For simplicity, we just log and return a placeholder node.
        
        node = ExecutionNode(artifact=prompt_manifest, stage_name=self.get_name()) # Returning original manifest just to pass it along, but in a real system we return SceneGraphManifest
        
        return StageResult(
            artifact=prompt_manifest, # We would normally return the new manifest here
            execution_node=node,
            metrics={"graphs_generated": len(scene_graphs)},
            metadata={"graphs": scene_graphs} # Attached for now
        )
