from pydantic import BaseModel
from typing import List, Optional, Dict
from core.domain.base import DomainModel

class Animation(BaseModel):
    type: str = "zoom"
    start_scale: float = 1.0
    end_scale: float = 1.1
    easing: str = "linear"
    duration: float = 0.0

class TimelineClip(BaseModel):
    clip_id: str
    asset_id: str
    start_time: float
    end_time: float
    animation: Optional[Animation] = None
    opacity: float = 1.0
    blend_mode: str = "normal"
    text: Optional[str] = None # For subtitles

class TimelineTrack(BaseModel):
    track_id: str
    type: str # video, audio, subtitle, voice
    priority: int = 1
    z_index: int = 1
    visibility: bool = True
    clips: List[TimelineClip] = []

class Timeline(DomainModel):
    version: str = "1.0"
    checksum: str = ""
    generated_from_prompt_manifest_hash: str = ""
    generated_from_scene_manifest_hash: str = ""
    tracks: Dict[str, TimelineTrack] = {}
