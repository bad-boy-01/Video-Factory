from pydantic import BaseModel
from typing import Optional

class RenderPreset(BaseModel):
    name: str = "lightning"
    width: int = 1024
    height: int = 1024
    steps: int = 4
    cfg: float = 0.0
    sampler: str = "euler"
    negative_prompt: str = ""
    
class ModelConfig(BaseModel):
    model_id: str = "runwayml/stable-diffusion-v1-5"
    adapter: Optional[str] = "ByteDance/SDXL-Lightning"
    vae: Optional[str] = None
    cache_dir: str = "/tmp/models"
    dtype: str = "float16"
    cpu_offload: bool = True
    
class RenderJob(BaseModel):
    prompt: str
    negative_prompt: str
    seed: int
    preset: RenderPreset
    # Future extensibility: lora, controlnet, etc.
