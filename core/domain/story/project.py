from pydantic import BaseModel
from typing import Optional
from core.domain.base import DomainModel
from core.domain.story.bible import StoryBible
from core.domain.story.style import StyleGuide
from core.domain.scene.manifest import SceneManifest
from core.domain.prompt.ast import PromptManifest
from core.domain.timeline.models import Timeline
from core.domain.assets.registry import AssetRegistry

class ProjectMetadata(BaseModel):
    project_name: str
    dataset_id: str
    author: str = ""
    target_resolution: str = "1920x1080"
    target_fps: int = 24
    
class ProjectManifest(DomainModel):
    metadata: ProjectMetadata
    source_text: str = ""
    
    # Derived manifests
    bible: Optional[StoryBible] = None
    style: Optional[StyleGuide] = None
    scenes: Optional[SceneManifest] = None
    prompts: Optional[PromptManifest] = None
    timeline: Optional[Timeline] = None
    registry: Optional[AssetRegistry] = None
