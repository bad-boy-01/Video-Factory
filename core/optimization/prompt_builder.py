from core.pipeline.stage import CompilerStage, StageResult
from core.domain.assets.execution import ExecutionNode
from core.domain.prompt.ast import (
    PromptManifest, PromptManifestEntry, PromptAST, CameraAST,
    SubjectAST, EnvironmentAST, LightingAST, CompositionAST, QualityAST, NegativeAST, CharacterAST
)
from core.domain.scene.manifest import SceneManifest, ShotManifest
from core.domain.story.bible import StoryBible
from core.pipeline.context import PipelineContext
from typing import Any
import hashlib

class PromptBuilderStage(CompilerStage):
    def get_name(self) -> str:
        return "PromptBuilderStage"

    def get_providers(self) -> list:
        return []
        
    def inputs(self, context: PipelineContext) -> list[Any]:
        inputs = []
        for node in context.execution_nodes:
            if isinstance(node.artifact, (SceneManifest, ShotManifest, StoryBible)):
                inputs.append(node.artifact)
        return inputs
        
    def outputs(self) -> list[str]:
        return ["prompt_manifest"]
        
    def generator_signature(self) -> str:
        return f"{self.get_name()}_ast_builder_v2.0"

    def execute(self, context: PipelineContext) -> StageResult:
        shot_manifest = None
        scene_manifest = None
        bible = None
        
        for node in context.execution_nodes:
            if isinstance(node.artifact, ShotManifest):
                shot_manifest = node.artifact
            elif isinstance(node.artifact, SceneManifest):
                scene_manifest = node.artifact
            elif isinstance(node.artifact, StoryBible):
                bible = node.artifact
                
        if not shot_manifest:
            raise ValueError("PromptBuilder requires ShotManifest.")
        if not scene_manifest:
            raise ValueError("PromptBuilder requires SceneManifest.")
            
        # Build scene lookup
        scene_map = {s.scene_id: s for s in scene_manifest.scenes}
            
        prompts = []
        for shot in shot_manifest.shots:
            # shot_id format: shot_{scene_hash}_{idx}
            parts = shot.shot_id.split("_")
            scene_hash = parts[1] if len(parts) > 1 else ""
            
            # Find matching scene
            matching_scene = None
            for s in scene_manifest.scenes:
                h = hashlib.sha256(s.scene_id.encode('utf-8')).hexdigest()[:8]
                if h == scene_hash:
                    matching_scene = s
                    break
                    
            location = matching_scene.location if matching_scene else "unknown location"
            time_of_day = "day" # Default
            weather = "clear" # Default
            if bible and location in bible.locations:
                loc_profile = bible.locations[location]
                time_of_day = loc_profile.time_defaults or time_of_day
                weather = loc_profile.weather_defaults or weather
                
            # Build Characters
            ast_chars = []
            for cast_member in shot.cast:
                char_ast = CharacterAST(
                    name=cast_member.character_id,
                    emotion=cast_member.emotion,
                    pose=cast_member.pose,
                    visibility=cast_member.visibility,
                    interaction=cast_member.interaction
                )
                if bible and cast_member.character_id in bible.characters:
                    profile = bible.characters[cast_member.character_id]
                    char_ast.appearance_tags = [
                        profile.appearance.hair, profile.appearance.eyes, profile.appearance.face, 
                        profile.appearance.age, profile.appearance.body
                    ]
                    char_ast.clothing_tags = [profile.appearance.clothing] + profile.appearance.color_palette
                    char_ast.signature_tags = profile.appearance.signature
                    char_ast.bindings = profile.bindings.model_dump()
                ast_chars.append(char_ast)
                
            # Deterministic seed based on hash
            seed_hash = hashlib.md5(shot.shot_id.encode('utf-8')).hexdigest()
            seed = int(seed_hash, 16) % (2**32 - 1)
            
            ast = PromptAST(
                subject=SubjectAST(description=f"{shot.purpose} shot emphasizing {shot.focus}"),
                characters=ast_chars,
                environment=EnvironmentAST(location=location, time_of_day=time_of_day, weather=weather),
                camera=CameraAST(
                    type=shot.camera_type,
                    lens=shot.lens,
                    angle=shot.angle,
                    distance=shot.distance,
                    movement=shot.movement
                ),
                lighting=LightingAST(style="cinematic lighting"),
                composition=CompositionAST(style="cinematic framing"),
                quality=QualityAST(tags=["masterpiece", "high resolution", "intricate detail"]),
                negative=NegativeAST(tags=["low quality", "blurry", "distorted", "bad anatomy", "watermark"])
            )
            
            from core.domain.prompt.visual_scene import VisualScene, VisualCharacter, VisualEnvironment, VisualCamera, VisualStyle
            
            # Construct VisualScene from AST and Bible state
            v_chars = []
            for cast_member in shot.cast:
                profile = bible.characters.get(cast_member.character_id) if bible else None
                if profile:
                    app_desc = f"Hair: {profile.appearance.hair}, Eyes: {profile.appearance.eyes}, Face: {profile.appearance.face}, Age: {profile.appearance.age}, Body: {profile.appearance.body}"
                    wardrobe_lock = matching_scene.state.wardrobe_locks.get(cast_member.character_id, "default") if matching_scene and matching_scene.state else "default"
                    # Lookup wardrobe
                    wardrobe_desc = ", ".join(bible.wardrobe.get(cast_member.character_id, {}).get(wardrobe_lock, [])) if bible and hasattr(bible, "wardrobe") else ""
                    bindings_list = [b.model_dump() for b in profile.bindings]
                else:
                    app_desc = "Unknown appearance"
                    wardrobe_desc = "Unknown wardrobe"
                    bindings_list = []
                    
                v_chars.append(VisualCharacter(
                    name=cast_member.character_id,
                    appearance=app_desc,
                    wardrobe=wardrobe_desc,
                    pose=cast_member.pose,
                    emotion=cast_member.emotion,
                    interaction=cast_member.interaction,
                    visibility=cast_member.visibility,
                    bindings=bindings_list
                ))
                
            state = matching_scene.state if matching_scene else None
            v_env = VisualEnvironment(
                location_desc=location,
                time=state.time if state else time_of_day,
                season=state.season if state else "",
                weather=state.weather if state else weather,
                lighting=state.lighting if state else "cinematic lighting",
                palette=state.palette if state else "",
                environment_state=state.environment_state if state else ""
            )
            
            v_cam = VisualCamera(
                type=shot.camera_type,
                lens=shot.lens,
                angle=shot.angle,
                distance=shot.distance,
                movement=shot.movement
            )
            
            v_style = VisualStyle(
                composition="cinematic framing",
                lighting_style=state.lighting if state else "cinematic lighting",
                color_grade=state.palette if state else "",
                mood=shot.emotion,
                quality_tags=["masterpiece", "high resolution", "intricate detail"],
                negative_tags=["low quality", "blurry", "distorted", "bad anatomy", "watermark"]
            )
            
            visual_scene = VisualScene(
                subject=f"{shot.purpose} shot emphasizing {shot.focus}",
                characters=v_chars,
                environment=v_env,
                camera=v_cam,
                style=v_style,
                props=[],
                vehicles=[]
            )
            
            entry = PromptManifestEntry(
                prompt_id=f"prompt_{seed_hash[:8]}",
                scene_id=matching_scene.scene_id if matching_scene else "unknown",
                shot_id=shot.shot_id,
                ast=ast,
                visual_scene=visual_scene,
                seed=seed
            )
            prompts.append(entry)
            
        manifest = PromptManifest(
            prompts=prompts,
            generator="PromptBuilderStage",
            generator_version="3.0.0"
        )
        
        node = ExecutionNode(artifact=manifest, stage_name="PromptBuilderStage")
        
        return StageResult(
            artifact=manifest,
            execution_node=node,
            metrics={"prompts_generated": len(prompts)},
            metadata={}
        )
