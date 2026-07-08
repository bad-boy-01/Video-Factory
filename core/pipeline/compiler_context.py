from typing import Optional, Any
from core.pipeline.context import PipelineContext
from core.domain.workspace import WorkspaceManager
from core.domain.assets.registry import AssetRegistry

class CompilerContext:
    """
    The formal runtime context provided to all stages.
    Contains the immutable pipeline data (PipelineContext) as well as the 
    managers for stateful components (Workspace, Registry, Queue, etc.).
    """
    def __init__(
        self, 
        pipeline_context: PipelineContext,
        workspace: WorkspaceManager,
        registry: AssetRegistry,
        queue: Optional[Any] = None # Will be RenderQueue
    ):
        self.pipeline = pipeline_context
        self.workspace = workspace
        self.registry = registry
        self.queue = queue
        
    def __getattr__(self, name: str) -> Any:
        return getattr(self.pipeline, name)
