from typing import Protocol, TypeVar, Generic, Mapping, Any
from dataclasses import dataclass
from .context import PipelineContext

T = TypeVar("T")

@dataclass(frozen=True)
class StageResult(Generic[T]):
    artifact: T
    execution_node: Any  # core.domain.asset.ExecutionNode
    metrics: Mapping[str, Any]
    metadata: Mapping[str, Any]


from enum import Enum
from abc import ABC, abstractmethod

class CachePolicy(Enum):
    ALWAYS = "ALWAYS"
    FINGERPRINT = "FINGERPRINT"
    NEVER = "NEVER"
    EXTERNAL = "EXTERNAL"

class CompilerStage(ABC):
    """
    A single stage in the AI compiler pipeline.
    """
    
    def get_name(self) -> str:
        """Returns the human-readable name of the stage."""
        return self.__class__.__name__
        
    def get_providers(self) -> list:
        """Returns a list of providers used by this stage, so the executor can manage their lifecycle."""
        return []
        
    def inputs(self, context: PipelineContext) -> list[Any]:
        """Returns the specific parent artifacts or data this stage depends on."""
        return []
        
    def outputs(self) -> list[str]:
        """Returns the artifact types or names this stage produces."""
        return []
        
    def generator_signature(self) -> str:
        """Returns the signature of the generator (e.g. Qwen1.5-4B + PromptBuilder v2.3)."""
        return "default_signature"
        
    def cache_policy(self) -> CachePolicy:
        """Returns the cache policy for this stage. Defaults to FINGERPRINT."""
        return CachePolicy.FINGERPRINT
        
    @abstractmethod
    def execute(self, context: PipelineContext) -> StageResult:
        """Executes the core logic of the stage."""
        pass

PipelineStage = CompilerStage  # Alias for backward compatibility
