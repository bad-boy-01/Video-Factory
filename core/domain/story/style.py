from pydantic import BaseModel
from typing import Optional
from core.domain.base import DomainModel

class StyleGuide(DomainModel):
    visual_style: str = "Korean Manhwa"
    render_quality: str = "masterpiece, best quality, ultra-detailed"
    camera_style: str = "cinematic, 8k resolution"
    lighting: str = "soft volumetric lighting, dramatic shadows"
    aspect_ratio: str = "16:9"
    negative_prompt: str = "low quality, bad anatomy, bad hands, missing fingers, worst quality, cropped, blurry, monochrome"
