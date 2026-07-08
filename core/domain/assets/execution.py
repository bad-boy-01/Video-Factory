from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import datetime
from pathlib import Path


@dataclass(frozen=True)
class ProvenanceRecord:
    """Immutable record of how an asset was generated."""
    model_id: str
    revision: str
    prompt_hash: str
    seed: int
    scheduler: str
    guidance_scale: float
    inference_steps: int
    timestamp: str = field(default_factory=lambda: datetime.datetime.utcnow().isoformat())
    diffusers_version: Optional[str] = None
    torch_version: Optional[str] = None


@dataclass
class GenerationRequest:
    """Immutable request object encapsulating all generation parameters."""
    compiled_prompt: str
    negative_prompt: str
    seed: int
    prompt_hash: str
    model_id: str
    output_path: Path
    width: int
    height: int
    steps: int
    guidance_scale: float


@dataclass
class GeneratedImage:
    """The central artifact for generated visual assets."""
    image_path: Path
    width: int
    height: int
    seed: int
    prompt_hash: str
    model_id: str
    cache_hit: bool
    provenance: Optional[ProvenanceRecord] = None


from pydantic import BaseModel

class FrameEntry(BaseModel):
    shot_id: str
    image_path: Path
    duration: float = 3.0

class FrameManifest(BaseModel):
    frames: List[FrameEntry] = []


@dataclass(frozen=True)
class ExecutionNode:
    """The canonical runtime object grouping an artifact with its provenance and generation request."""
    artifact: Any
    request: Optional[GenerationRequest] = None
    provenance: Optional[ProvenanceRecord] = None
    cache_key: str = ""
    stage_name: str = ""
    execution_time: float = 0.0
    retry_count: int = 0
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    contract_results: List[Any] = field(default_factory=list)  # List[ContractResult]
    timestamp: str = field(default_factory=lambda: datetime.datetime.utcnow().isoformat())



