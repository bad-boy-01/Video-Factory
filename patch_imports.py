import os

replacements = {
    "core.domain.bible": "core.domain.story.bible",
    "core.domain.style": "core.domain.story.style",
    "core.domain.scene": "core.domain.scene.manifest",
    "core.domain.timeline": "core.domain.timeline.models",
    "core.domain.prompt": "core.domain.prompt.ast",
    "core.domain.asset": "core.domain.assets.execution",
    "core.domain.registry": "core.domain.assets.registry",
    "core.domain.project": "core.domain.story.project",
    
    "ai.reasoning.scene_splitter": "core.planning.scene_splitter",
    "ai.reasoning.story_bible": "core.planning.story_bible_stage",
    "ai.planning.shot_planner": "core.planning.shot_planner",
    "ai.planning.camera_planner": "core.planning.camera_planner",
    "ai.planning.timeline_builder": "core.planning.timeline_builder",
    "ai.planning.validator": "core.validation.pipeline_validator",
    "ai.prompting.prompt_stage": "core.optimization.prompt_builder",
    "core.pipeline.render_queue": "core.rendering.render_queue",
    "ai.generation.image_stage": "core.rendering.image_stage",
    "ai.generation.rendering_stage": "core.rendering.assembly_stage",
    "core.pipeline.executor": "core.rendering.executor"
}

for root, dirs, files in os.walk("."):
    if ".git" in root or ".cache" in root or "workspace" in root:
        continue
    for file in files:
        if file.endswith(".py"):
            path = os.path.join(root, file)
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            
            new_content = content
            for old, new in replacements.items():
                new_content = new_content.replace("from " + old, "from " + new)
                new_content = new_content.replace("import " + old, "import " + new)
                
            if new_content != content:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(new_content)
                print(f"Patched {path}")
