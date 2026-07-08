from pydantic import BaseModel
from typing import List
from core.domain.base import DomainModel

class SceneMemory(DomainModel):
    scene_id: str
    current_location: str = ""
    current_weather: str = ""
    time_of_day: str = ""
    active_characters: List[str] = []
    important_objects: List[str] = []
    mood: str = ""
    lighting: str = ""
