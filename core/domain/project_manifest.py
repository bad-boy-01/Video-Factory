from pydantic import Field
from core.domain.base import DomainModel
import datetime
from core.compiler_version import PIPELINE_VERSION

class ProjectManifest(DomainModel):
    """
    Defines the root configuration for a project, tracking the core inputs and default versions.
    """
    project_name: str = "default_project"
    created_at: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())
    pipeline_version: str = PIPELINE_VERSION
    default_model: str = "sdxl-lightning"
    default_preset: str = "fast"
    novel_hash: str = ""
    workspace_version: str = "1"
    source_text: str = ""
