from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from core.domain.base import DomainModel

class BindingReference(BaseModel):
    id: str
    kind: str # e.g. 'lora', 'ip_adapter', 'embedding', 'character_reference'
    priority: int = 1
    weight: float = 1.0
    required: bool = True

class Appearance(BaseModel):
    hair: str = ""
    eyes: str = ""
    face: str = ""
    age: str = ""
    body: str = ""
    clothing: str = ""
    color_palette: List[str] = Field(default_factory=list)
    signature: List[str] = Field(default_factory=list)

class CharacterVisualProfile(BaseModel):
    id: str
    name: str
    appearance: Appearance = Field(default_factory=Appearance)
    bindings: List[BindingReference] = Field(default_factory=list)

class Location(BaseModel):
    id: str
    name: str
    appearance: str = ""
    architecture: str = ""
    weather_defaults: str = ""
    time_defaults: str = ""
    lighting_presets: str = ""

class Organization(BaseModel):
    id: str
    name: str
    description: str = ""
    uniforms: str = ""
    symbols: List[str] = Field(default_factory=list)

class Prop(BaseModel):
    id: str
    name: str
    description: str = ""
    visual_details: str = ""

class Vehicle(BaseModel):
    id: str
    name: str
    description: str = ""
    visual_details: str = ""

class TimelineEvent(BaseModel):
    event_id: str
    description: str = ""
    timestamp: str = ""

class Relationship(BaseModel):
    char1_id: str
    char2_id: str
    nature: str = ""
    dynamic: str = ""

class StoryBible(DomainModel):
    version: int = 4
    characters: Dict[str, CharacterVisualProfile] = Field(default_factory=dict)
    locations: Dict[str, Location] = Field(default_factory=dict)
    organizations: Dict[str, Organization] = Field(default_factory=dict)
    props: Dict[str, Prop] = Field(default_factory=dict)
    vehicles: Dict[str, Vehicle] = Field(default_factory=dict)
    wardrobe: Dict[str, Dict[str, List[str]]] = Field(default_factory=dict) # char_id -> outfit_name -> List of items
    visual_themes: List[str] = Field(default_factory=list)
    symbolism: Dict[str, str] = Field(default_factory=dict)
    timeline: List[TimelineEvent] = Field(default_factory=list)
    relationships: List[Relationship] = Field(default_factory=list)
    style_guide: Dict[str, str] = Field(default_factory=dict)
    production_notes: str = ""
