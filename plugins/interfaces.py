from __future__ import annotations
from typing import Protocol, Dict, Any, List, TYPE_CHECKING, Callable
from pathlib import Path
from pydantic import BaseModel

if TYPE_CHECKING:
    from PIL import Image
    from core.domain.rendering.presets import RenderJob
from core.pipeline.context import PipelineContext

class LLMProvider(Protocol):
    def initialize(self) -> None:
        ...
    def load(self) -> None:
        ...
    def unload(self) -> None:
        ...
    def shutdown(self) -> None:
        ...
    def generate_json(self, prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Generates a structured JSON response from the LLM, guaranteed to match schema."""
        ...

from dataclasses import dataclass

@dataclass
class DiffusionConfig:
    model_id: str = "runwayml/stable-diffusion-v1-5"
    cache_dir: str = "/tmp/models"
    revision: str = "main"
    dtype: str = "float16"  # resolved to a real torch.dtype only inside providers that need torch
    cpu_offload: bool = True
    adapter: str | None = None

class ProviderHealth(BaseModel):
    loaded: bool
    device: str
    model: str
    dtype: str
    vram_allocated_gb: float

class ProviderCapability(BaseModel):
    modality: str = "image"
    max_resolution: tuple[int, int] = (2048, 2048)
    supports_lora: bool = False
    supports_controlnet: bool = False
    supports_img2img: bool = False
    supports_ip_adapter: bool = False
    supports_inpainting: bool = False

class PromptFingerprint(BaseModel):
    provider_name: str
    provider_version: str
    prompt_hash: str
    model_hash: str
    sampling_hash: str
    schema_hash: str
    
    @property
    def key(self) -> str:
        import hashlib
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()

from core.domain.prompt.render_plan import RenderPlan
from core.domain.prompt.provider_request import ProviderRequest

class ProviderCompiler(Protocol):
    def compile_plan(self, plan: RenderPlan) -> ProviderRequest:
        """Compiles a provider-agnostic RenderPlan into a backend-specific ProviderRequest."""
        ...

class ImageGenerationProvider(Protocol):
    def generate(self, request: ProviderRequest, callback: Callable[[int, int], None] = None) -> Image.Image:
        """Executes a fully compiled ProviderRequest."""
        ...
        
    def capabilities(self) -> ProviderCapability:
        ...
        
    def warmup(self) -> None:
        ...
        
    def load(self) -> None:
        ...
        
    def health_check(self) -> ProviderHealth:
        ...
        
    def unload(self) -> None:
        ...

class VideoRendererProvider(Protocol):
    def render_video(self, manifest: 'FrameManifest', audio_paths: List[Path], output_path: Path) -> Path:
        """Assembles frames from a manifest into a final video."""
        ...
