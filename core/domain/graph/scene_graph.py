from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Literal
from core.domain.base import DomainModel

NodeType = Literal["Character", "Prop", "Vehicle", "Environment", "Light", "Camera"]

class GraphNode(BaseModel):
    id: str
    type: NodeType
    properties: Dict[str, str] = Field(default_factory=dict)

class GraphEdge(BaseModel):
    source_id: str
    target_id: str
    relationship: str # e.g., 'holding', 'looking_at', 'standing_on', 'inside', 'wearing', 'illuminates'

class SceneGraph(DomainModel):
    nodes: List[GraphNode] = Field(default_factory=list)
    edges: List[GraphEdge] = Field(default_factory=list)
