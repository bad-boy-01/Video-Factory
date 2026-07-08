from pydantic import BaseModel, Field
from typing import List, Dict, Literal
from core.domain.base import DomainModel

BeatRelationship = Literal["causes", "escalates", "reveals", "resolves", "contrasts", "parallels"]

class BeatNode(BaseModel):
    beat_id: str
    description: str
    emotion: str
    importance: str

class BeatEdge(BaseModel):
    source_id: str
    target_id: str
    relationship: BeatRelationship

class BeatGraph(DomainModel):
    scene_id: str
    nodes: List[BeatNode] = Field(default_factory=list)
    edges: List[BeatEdge] = Field(default_factory=list)
