from typing import List, Optional, Dict
from pydantic import BaseModel
from core.domain.base import DomainModel

class SceneDirectives(BaseModel):
    keep_same_outfit: bool = True
    keep_camera_style: bool = False
    emphasize_background: bool = False
    allow_creativity: str = "medium"
    priority: str = "character"

class VisualScreenplay(BaseModel):
    mood: str = ""
    emotion: str = ""
    pacing: str = "medium"
    lighting: str = ""
    camera: str = ""
    lens: str = ""
    composition: str = ""
    characters: List[str] = []
    focus: str = ""
    transition: str = "cut"
    duration_seconds: float = 4.0

class Beat(DomainModel):
    text: str
    visual_screenplay: VisualScreenplay = VisualScreenplay()

class Scene(DomainModel):
    beats: List[Beat] = []
    location_id: Optional[str] = None
    time_of_day: str = "day"
    weather: str = "clear"
    directives: SceneDirectives = SceneDirectives()

class Chapter(DomainModel):
    title: str
    scenes: List[Scene] = []

class Novel(DomainModel):
    title: str
    chapters: List[Chapter] = []
