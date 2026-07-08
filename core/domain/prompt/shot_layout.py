from pydantic import BaseModel, Field
from typing import Dict, List
from core.domain.base import DomainModel

class ShotLayout(DomainModel):
    shot_id: str
    subject_positions: Dict[str, str] = Field(default_factory=dict) # e.g., "John": "center foreground"
    screen_occupancy: Dict[str, str] = Field(default_factory=dict) # e.g., "John": "30%"
    eye_direction: Dict[str, str] = Field(default_factory=dict)
    camera_axis: str = ""
    rule_of_thirds: List[str] = Field(default_factory=list)
    negative_space: str = ""
    foreground: List[str] = Field(default_factory=list)
    midground: List[str] = Field(default_factory=list)
    background: List[str] = Field(default_factory=list)
