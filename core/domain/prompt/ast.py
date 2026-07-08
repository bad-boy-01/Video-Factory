from pydantic import BaseModel
from typing import List, Optional
from core.domain.base import DomainModel

class SubjectAST(BaseModel):
    description: str = ""

class CharacterAST(BaseModel):
    name: str = ""
    appearance_tags: List[str] = []
    clothing_tags: List[str] = []
    signature_tags: List[str] = []
    action: str = ""
    emotion: str = ""
    pose: str = ""
    visibility: str = ""
    interaction: str = ""
    bindings: dict = {}

class EnvironmentAST(BaseModel):
    location: str = ""
    time_of_day: str = ""
    weather: str = ""

class CameraAST(BaseModel):
    type: str = ""
    lens: str = ""
    angle: str = ""
    distance: str = ""
    movement: str = ""

class LightingAST(BaseModel):
    style: str = ""

class CompositionAST(BaseModel):
    style: str = ""

class MoodAST(BaseModel):
    mood: str = ""

class QualityAST(BaseModel):
    tags: List[str] = []

class TechnicalAST(BaseModel):
    steps: int = 25
    cfg: float = 7.0
    sampler: str = "Euler a"
    width: int = 1024
    height: int = 1024

class NegativeAST(BaseModel):
    tags: List[str] = []

class PromptAST(BaseModel):
    subject: SubjectAST = SubjectAST()
    characters: List[CharacterAST] = []
    environment: EnvironmentAST = EnvironmentAST()
    camera: CameraAST = CameraAST()
    lighting: LightingAST = LightingAST()
    composition: CompositionAST = CompositionAST()
    mood: MoodAST = MoodAST()
    quality: QualityAST = QualityAST()
    technical: TechnicalAST = TechnicalAST()
    negative: NegativeAST = NegativeAST()

from core.domain.prompt.visual_scene import VisualScene

class PromptManifestEntry(BaseModel):
    prompt_id: str
    scene_id: str
    shot_id: str
    ast: PromptAST
    visual_scene: Optional[VisualScene] = None
    seed: int
    model_target: str = "sdxl"
    
class PromptManifest(DomainModel):
    prompts: List[PromptManifestEntry] = []
