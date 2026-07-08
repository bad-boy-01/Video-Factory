from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from core.domain.base import DomainModel

class LogicalRenderPlan(BaseModel):
    subject: str = ""
    framing: str = ""
    emphasis: str = ""
    mood: str = ""

class PhysicalRenderPlan(BaseModel):
    width: int = 1024
    height: int = 1024
    steps: int = 25
    cfg: float = 7.0
    seed: int = 0
    bindings: List[Dict[str, str]] = Field(default_factory=list)
    loras: List[str] = Field(default_factory=list)
    controlnets: Dict[str, Any] = Field(default_factory=dict)

class RenderPlan(DomainModel):
    shot_id: str
    logical: LogicalRenderPlan = Field(default_factory=LogicalRenderPlan)
    physical: PhysicalRenderPlan = Field(default_factory=PhysicalRenderPlan)
