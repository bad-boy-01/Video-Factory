from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from core.domain.base import DomainModel

class StyleProfile(BaseModel):
    name: str
    camera: Dict[str, str] = Field(default_factory=dict)
    lighting: Dict[str, str] = Field(default_factory=dict)
    composition: Dict[str, str] = Field(default_factory=dict)
    color: Dict[str, str] = Field(default_factory=dict)
    editing: Dict[str, str] = Field(default_factory=dict)
    motion: Dict[str, str] = Field(default_factory=dict)
    lens: Dict[str, str] = Field(default_factory=dict)

class StyleStack(BaseModel):
    base_cinematic: Optional[StyleProfile] = None
    project_style: Optional[StyleProfile] = None
    scene_style: Optional[StyleProfile] = None
    beat_override: Optional[StyleProfile] = None

class FrameInstruction(BaseModel):
    image_asset_id: Optional[str] = None
    transition: Optional[str] = None
    duration: float = 2.0

class CastMember(BaseModel):
    character_id: str
    emotion: str = "neutral"
    pose: str = "standing"
    visibility: str = "foreground"
    interaction: str = "none"

class Shot(BaseModel):
    shot_id: str
    beat_id: str = ""  # Links this shot back to the narration beat it belongs to

    # Semantic Intent (From ShotPlanner)
    purpose: str = ""
    emotion: str = ""
    importance: str = ""
    focus: str = ""
    
    # Style Data (From DirectorPlanner)
    style: str = ""
    style_stack: Optional[StyleStack] = None
    
    # Physical Camera (From CameraPlanner)
    camera_type: str = ""
    lens: str = ""
    angle: str = ""
    distance: str = ""
    movement: str = ""
    
    # Cast Data (From CastPlanner)
    cast: List[CastMember] = []
    
    duration: float = 2.0
    frames: List[FrameInstruction] = []

class Beat(BaseModel):
    beat_id: str
    description: str
    emotion: str
    shots: List[Shot] = []

class SceneState(BaseModel):
    time: str = ""
    season: str = ""
    weather: str = ""
    lighting: str = ""
    palette: str = ""
    wardrobe_locks: Dict[str, str] = Field(default_factory=dict)
    hair_state: Dict[str, str] = Field(default_factory=dict)
    damage_state: Dict[str, str] = Field(default_factory=dict)
    prop_state: Dict[str, str] = Field(default_factory=dict)
    vehicle_state: Dict[str, str] = Field(default_factory=dict)
    environment_state: str = ""

class Scene(BaseModel):
    scene_id: str
    chapter: int
    start_offset: int
    end_offset: int
    estimated_duration: float
    characters: List[str]
    location: str
    emotion: str
    beats: List[Beat] = []
    state: Optional[SceneState] = None

class SceneManifest(DomainModel):
    scenes: List[Scene] = []

class ShotManifest(DomainModel):
    shots: List[Shot] = []
