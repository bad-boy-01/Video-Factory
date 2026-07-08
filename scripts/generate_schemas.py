import os
import json
from pathlib import Path

# Correct imports
from core.domain.story.bible import StoryBible
from core.domain.scene.manifest import SceneManifest
from core.domain.graph.beat_graph import BeatGraph
from core.domain.graph.scene_graph import SceneGraph
from core.domain.prompt.render_plan import RenderPlan
from core.domain.prompt.provider_request import ProviderRequest
from core.domain.provider_capability import ProviderCapability
from core.compiler_version import SCHEMA_VERSION

def generate_schemas():
    models = {
        "StoryBible": StoryBible,
        "SceneManifest": SceneManifest,
        "BeatGraph": BeatGraph,
        "SceneGraph": SceneGraph,
        "RenderPlan": RenderPlan,
        "ProviderRequest": ProviderRequest,
        "ProviderCapability": ProviderCapability
    }
    
    schemas_dir = Path("schemas")
    schemas_dir.mkdir(exist_ok=True)
    
    for name, model_cls in models.items():
        try:
            schema = model_cls.model_json_schema()
        except AttributeError:
            schema = model_cls.schema()
            
        filename = schemas_dir / f"{name}_v{SCHEMA_VERSION}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(schema, f, indent=2)
        print(f"Generated {filename}")

if __name__ == "__main__":
    generate_schemas()
