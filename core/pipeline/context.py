from typing import Dict, Optional, List, Any
from uuid import UUID
from pydantic import BaseModel, Field

from core.domain.story.project import ProjectManifest
from core.domain.story.bible import StoryBible
from core.domain.assets.execution import ExecutionNode
from core.domain.prompt.ast import PromptAST

class PipelineContext(BaseModel):
    """
    The central state object that flows through the execution pipeline.
    Stages receive this read-only context. It is never mutated.
    Instead, a ContextReducer generates a new instance.
    """
    model_config = {"frozen": True}

    project_manifest: ProjectManifest
    story_bible: Optional[StoryBible] = None

    # Active scope for the current execution batch
    # Using Any to avoid importing the wrong Scene class from core.domain.story
    # (which conflicts with core.domain.scene.manifest.Scene used by pipeline stages)
    current_chapter: Optional[Any] = None
    current_scene: Optional[Any] = None

    # Tracked entities generated during execution
    execution_nodes: List[ExecutionNode] = Field(default_factory=list)
    prompts: Dict[UUID, PromptAST] = Field(default_factory=dict)

    # Ephemeral state for inter-stage communication of non-domain data
    state: dict = Field(default_factory=dict)
