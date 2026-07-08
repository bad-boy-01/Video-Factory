from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from core.domain.base import DomainModel

class WorldState(BaseModel):
    environment_conditions: Dict[str, str] = Field(default_factory=dict)
    weather: str = ""
    time_of_day: str = ""
    lighting_conditions: str = ""
    prop_states: Dict[str, str] = Field(default_factory=dict)
    vehicle_states: Dict[str, str] = Field(default_factory=dict)
    destroyed_elements: List[str] = Field(default_factory=list)

class CharacterState(BaseModel):
    character_id: str
    current_emotion: str = "neutral"
    injuries: List[str] = Field(default_factory=list)
    clothing_state: Dict[str, str] = Field(default_factory=dict) # e.g. coat: removed, shirt: torn
    held_items: List[str] = Field(default_factory=list)
    current_location: str = ""
    posture: str = "standing"

class CameraState(BaseModel):
    last_shot_type: str = ""
    last_lens: str = ""
    last_angle: str = ""
    last_movement: str = ""
    camera_position: str = ""
    camera_direction: str = ""

class ShotState(BaseModel):
    shot_id: str
    camera_position: str = ""
    camera_direction: str = ""
    subject_position: Dict[str, str] = Field(default_factory=dict) # character_id -> position
    character_pose: Dict[str, str] = Field(default_factory=dict) # character_id -> pose
    eye_direction: Dict[str, str] = Field(default_factory=dict) # character_id -> direction
    prop_locations: Dict[str, str] = Field(default_factory=dict) # prop_id -> location
    focus_target: str = ""
    depth_of_field: str = ""
    animation_seed: int = 0
