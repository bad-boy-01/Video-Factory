from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class VisualCharacter(BaseModel):
    name: str
    appearance: str
    wardrobe: str
    pose: str
    emotion: str
    interaction: str
    visibility: str
    bindings: List[Dict[str, str]] = Field(default_factory=list) # List of flattened binding maps

class VisualEnvironment(BaseModel):
    location_desc: str
    time: str
    season: str
    weather: str
    lighting: str
    palette: str
    environment_state: str

class VisualCamera(BaseModel):
    type: str
    lens: str
    angle: str
    distance: str
    movement: str

class VisualStyle(BaseModel):
    composition: str
    lighting_style: str
    color_grade: str
    mood: str
    quality_tags: List[str] = Field(default_factory=list)
    negative_tags: List[str] = Field(default_factory=list)

class VisualScene(BaseModel):
    """
    Zero-ID intermediate representation. 
    Everything is fully resolved into natural language or raw asset descriptors.
    """
    subject: str
    characters: List[VisualCharacter] = Field(default_factory=list)
    environment: VisualEnvironment
    camera: VisualCamera
    style: VisualStyle
    props: List[str] = Field(default_factory=list)
    vehicles: List[str] = Field(default_factory=list)
    
    # Render overrides
    width: int = 1024
    height: int = 1024
    steps: int = 25
    cfg: float = 7.0
    sampler: str = "Euler a"
