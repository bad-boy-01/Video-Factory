from pydantic import Field
from core.domain.base import DomainModel

class ProviderCapability(DomainModel):
    """
    Defines the capabilities of a specific provider, ensuring that the RenderGraph
    knows what nodes can be safely executed with it.
    """
    modality: str = "image"
    supports_controlnet: bool = False
    supports_ip_adapter: bool = False
    supports_lora: bool = False
    supports_img2img: bool = False
    max_resolution: tuple[int, int] = (1024, 1024)
