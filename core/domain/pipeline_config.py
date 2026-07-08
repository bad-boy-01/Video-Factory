from pydantic import BaseModel, Field
from typing import Optional
from core.domain.base import DomainModel

class PipelineConfig(DomainModel):
    project_id: str = "default_project"
    planning_model: str = "gpt-4o"
    llm_model: str = "Qwen/Qwen1.5-4B-Chat"
    diffusion_model: str = "stabilityai/stable-diffusion-xl-base-1.0"
    render_preset: str = "fast"
    scheduler: str = "euler_a"
    cache: bool = True
    batch_size: int = 4  # shots sharing resolution/steps/cfg/character-set are generated together
    seed: int = 42
    resume: bool = True
    num_workers: int = 1
    precision: str = "fp16"
    dtype: str = "float16"
    cache_dir: str = "/tmp/models"
    cpu_offload: bool = True
    use_face_id: bool = False  # opt-in: stronger facial identity via insightface + IP-Adapter-FaceID,
                                # alongside (not replacing) the standard IP-Adapter reference conditioning
