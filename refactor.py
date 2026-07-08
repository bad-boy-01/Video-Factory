import os
import shutil

def mkdirs(paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)

mkdirs([
    "core/domain/story",
    "core/domain/scene",
    "core/domain/timeline",
    "core/domain/prompt",
    "core/domain/render",
    "core/domain/assets",
    "core/planning",
    "core/validation",
    "core/optimization",
    "core/rendering"
])

# Move domain models
domain_moves = {
    "core/domain/bible.py": "core/domain/story/bible.py",
    "core/domain/style.py": "core/domain/story/style.py",
    "core/domain/scene.py": "core/domain/scene/manifest.py",
    "core/domain/timeline.py": "core/domain/timeline/models.py",
    "core/domain/prompt.py": "core/domain/prompt/ast.py",
    "core/domain/asset.py": "core/domain/assets/execution.py",
    "core/domain/registry.py": "core/domain/assets/registry.py",
    "core/domain/project.py": "core/domain/story/project.py",
}

# Move pipeline stages
stage_moves = {
    "ai/reasoning/scene_splitter.py": "core/planning/scene_splitter.py",
    "ai/reasoning/story_bible.py": "core/planning/story_bible_stage.py",
    "ai/planning/shot_planner.py": "core/planning/shot_planner.py",
    "ai/planning/camera_planner.py": "core/planning/camera_planner.py",
    "ai/planning/timeline_builder.py": "core/planning/timeline_builder.py",
    
    "ai/planning/validator.py": "core/validation/pipeline_validator.py",
    
    "ai/prompting/prompt_stage.py": "core/optimization/prompt_builder.py",
    
    "core/pipeline/render_queue.py": "core/rendering/render_queue.py",
    "ai/generation/image_stage.py": "core/rendering/image_stage.py",
    "ai/generation/rendering_stage.py": "core/rendering/assembly_stage.py",
    
    "core/pipeline/executor.py": "core/rendering/executor.py",
}

for src, dst in {**domain_moves, **stage_moves}.items():
    if os.path.exists(src):
        shutil.move(src, dst)
        print(f"Moved {src} to {dst}")
    else:
        print(f"WARN: {src} not found")
        
# Remove __init__ files so folders can be deleted
for root, dirs, files in os.walk("ai"):
    for file in files:
        if file == "__init__.py":
            os.remove(os.path.join(root, file))
            
# Clean up old ai folder
if os.path.exists("ai"):
    for root, dirs, files in os.walk("ai", topdown=False):
        for name in dirs:
            try:
                os.rmdir(os.path.join(root, name))
            except:
                pass
    try:
        os.rmdir("ai")
    except:
        pass
