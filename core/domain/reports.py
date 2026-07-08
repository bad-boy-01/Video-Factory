from pydantic import BaseModel, Field
from typing import Dict, List, Any

class DoctorReport(BaseModel):
    environment: Dict[str, str] = Field(default_factory=dict)
    checks: Dict[str, str] = Field(default_factory=dict)
    overall_status: str = "UNKNOWN"

class BenchmarkReport(BaseModel):
    planning: Dict[str, float] = Field(default_factory=dict)
    rendering: Dict[str, Any] = Field(default_factory=dict)
    assembly_time: float = 0.0
    cache: Dict[str, str] = Field(default_factory=dict)
    vram: Dict[str, float] = Field(default_factory=dict)
    assets: Dict[str, int] = Field(default_factory=dict)
    llm: Dict[str, Any] = Field(default_factory=dict)
