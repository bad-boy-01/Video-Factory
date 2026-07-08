from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Literal, Any
from core.domain.base import DomainModel
import datetime

class GenerationParams(BaseModel):
    resolution: tuple[int, int] = (1024, 1024)
    seed: int = 0
    steps: int = 25
    cfg: float = 7.0
    scheduler: str = "euler_a"
    frames: Optional[int] = None # For video
    fps: Optional[int] = None # For video

class ConditioningParams(BaseModel):
    prompt: str = ""
    negative_prompt: str = ""
    controlnets: Dict[str, Any] = Field(default_factory=dict) # type -> image tensor/path
    ip_adapter: Dict[str, Any] = Field(default_factory=dict) # character_id -> image tensor/path
    reference_images: List[Any] = Field(default_factory=list)
    masks: Dict[str, Any] = Field(default_factory=dict)

class BindingParams(BaseModel):
    loras: List[str] = Field(default_factory=list)
    embeddings: List[str] = Field(default_factory=list)
    style_models: List[str] = Field(default_factory=list)

class PostProcessParams(BaseModel):
    upscale: bool = False
    restore_faces: bool = False
    interpolation: bool = False

class ProviderMetadata(BaseModel):
    project_id: str = ""
    scene_id: str = ""
    shot_id: str = ""
    render_id: str = ""
    pipeline_version: str = "2.1"
    provider_name: str = ""
    created_at: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())
    fingerprint: str = ""

class ProviderRequest(DomainModel):
    request_type: Literal["image", "video"] = "image"
    generation: GenerationParams = Field(default_factory=GenerationParams)
    conditioning: ConditioningParams = Field(default_factory=ConditioningParams)
    bindings: BindingParams = Field(default_factory=BindingParams)
    postprocess: PostProcessParams = Field(default_factory=PostProcessParams)
    metadata: ProviderMetadata = Field(default_factory=ProviderMetadata)
