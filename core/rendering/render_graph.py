from typing import Any, Dict, List, Protocol, Type
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

class RenderArtifact(BaseModel):
    kind: str # e.g. "PROMPT", "IMAGE", "UPSCALE", "PROVIDER_REQUEST"
    data: Any

class RenderNode(Protocol):
    def get_name(self) -> str:
        ...
        
    def execute(self, inputs: Dict[str, RenderArtifact], config: Dict[str, Any]) -> Dict[str, RenderArtifact]:
        ...

class NodeRegistry:
    _nodes: Dict[str, Type[RenderNode]] = {}

    @classmethod
    def register(cls, name: str, node_cls: Type[RenderNode]):
        cls._nodes[name] = node_cls

    @classmethod
    def get_node(cls, name: str) -> Type[RenderNode]:
        if name not in cls._nodes:
            raise ValueError(f"RenderNode '{name}' is not registered.")
        return cls._nodes[name]

class RenderGraph:
    def __init__(self):
        self.nodes: List[RenderNode] = []
        
    def add_node(self, node: RenderNode):
        self.nodes.append(node)
        
    def add_node_by_name(self, name: str, **kwargs):
        node_cls = NodeRegistry.get_node(name)
        self.nodes.append(node_cls(**kwargs))
        
    def execute(self, initial_artifacts: Dict[str, RenderArtifact], config: Dict[str, Any]) -> Dict[str, RenderArtifact]:
        current_state = dict(initial_artifacts)
        for node in self.nodes:
            logger.info(f"Executing RenderNode: {node.get_name()}")
            try:
                outputs = node.execute(current_state, config)
                if outputs:
                    current_state.update(outputs)
            except Exception as e:
                logger.error(f"RenderNode {node.get_name()} failed: {e}")
                raise e
        return current_state
