from pydantic import BaseModel, Field
from uuid import UUID, uuid4
from datetime import datetime
from typing import Optional

class ProvenanceGraph(BaseModel):
    generated_from: Optional[UUID] = None
    model: Optional[str] = None
    model_revision: Optional[str] = None
    prompt_hash: Optional[str] = None
    story_bible_hash: Optional[str] = None
    config_hash: Optional[str] = None
    seed: Optional[int] = None

from typing import Generic, TypeVar, Optional, List, Dict, Any
from enum import Enum

class ArtifactState(Enum):
    CREATED = "CREATED"
    VALIDATED = "VALIDATED"
    STALE = "STALE"
    INVALID = "INVALID"
    ARCHIVED = "ARCHIVED"

class ArtifactMetadata(BaseModel):
    artifact_id: str = Field(default_factory=lambda: str(uuid4()))
    artifact_type: str
    
    fingerprint: str = ""
    dependency_hash: str = ""
    
    generator_signature: str = ""
    source_stage: str = ""
    
    schema_version: str = "1.0"
    pipeline_version: str = "0.4.5"
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    compile_duration_ms: int = 0
    
    parents: List[str] = Field(default_factory=list)
    children: List[str] = Field(default_factory=list)
    
    state: ArtifactState = ArtifactState.CREATED

T = TypeVar("T")

class CompilerArtifact(BaseModel, Generic[T]):
    metadata: ArtifactMetadata
    data: T

class DomainModel(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=datetime.utcnow)
