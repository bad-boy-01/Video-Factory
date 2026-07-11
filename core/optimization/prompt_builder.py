from core.pipeline.stage import CompilerStage, StageResult
from core.domain.assets.execution import ExecutionNode
from core.domain.prompt.ast import (
    PromptManifest, PromptManifestEntry, PromptAST, CameraAST,
    SubjectAST, EnvironmentAST, LightingAST, CompositionAST, QualityAST, NegativeAST, CharacterAST
)
from core.domain.scene.manifest import SceneManifest, ShotManifest, Beat
from core.domain.story.bible import StoryBible
from core.pipeline.context import PipelineContext
from typing import Any, Dict, Optional
import hashlib

# ---------------------------------------------------------------------------
# Director style → prompt tokens
# ---------------------------------------------------------------------------
STYLE_TOKENS: Dict[str, Dict[str, str]] = {
    "villeneuve": {
        "positive": "cinematic film still, anamorphic lens, soft diffuse light, high contrast silhouettes, "
                    "symmetrical composition, vast scale, desaturated palette, photorealistic, 8k",
        "negative": "anime, cartoon, sketch, watermark, text, low quality, blurry, distorted, bad anatomy",
    },
    "webtoon": {
        "positive": "Korean manhwa style, webtoon art, cel-shaded, vibrant saturated colors, clean lineart, "
                    "dynamic composition, bold outlines, flat shading, high resolution",
        "negative": "3d render, photorealistic, photography, low quality, blurry, distorted, bad anatomy, watermark, text",
    },
}
_DEFAULT_STYLE = "villeneuve"

# Camera distance → natural language framing tag
DISTANCE_TAGS = {
    "wide shot":         "wide establishing shot",
    "medium shot":       "medium shot",
    "close-up":          "close-up shot",
    "extreme close-up":  "extreme close-up macro shot",
}
ANGLE_TAGS = {
    "low angle":   "low angle shot",
    "high angle":  "high angle shot",
    "eye-level":   "eye-level shot",
}


def _build_character_tags(cast_member, bible: Optional[StoryBible]) -> str:
    """Return a comma-separated tag string describing a character's appearance."""
    tags = []
    if bible and cast_member.character_id in bible.characters:
        profile = bible.characters[cast_member.character_id]
        app = profile.appearance
        if app.hair:     tags.append(app.hair)
        if app.eyes:     tags.append(app.eyes)
        if app.face:     tags.append(app.face)
        if app.age:      tags.append(app.age)
        if app.body:     tags.append(app.body)
        if app.clothing: tags.append(app.clothing)
        tags.extend(app.color_palette)
    if cast_member.emotion and cast_member.emotion not in ("neutral", ""):
        tags.append(f"{cast_member.emotion} expression")
    if cast_member.pose and cast_member.pose not in ("standing", ""):
        tags.append(cast_member.pose)
    return ", ".join(t for t in tags if t)


def _build_prompt(
    shot,
    beat: Optional[Beat],
    scene,
    bible: Optional[StoryBible],
    style_name: str,
) -> str:
    """Assemble the full positive prompt string for one shot."""
    style = STYLE_TOKENS.get(style_name, STYLE_TOKENS[_DEFAULT_STYLE])
    parts = []

    # 1. Art / rendering style (highest weight anchor)
    parts.append(style["positive"])

    # 2. Characters — appearance, pose, emotion
    char_parts = []
    for cm in shot.cast:
        char_tags = _build_character_tags(cm, bible)
        if char_tags:
            char_parts.append(char_tags)
    if char_parts:
        parts.append(", ".join(char_parts))

    # 3. Camera — framing, angle, movement
    cam_tags = []
    dist_tag = DISTANCE_TAGS.get(shot.distance, shot.distance)
    if dist_tag: cam_tags.append(dist_tag)
    angle_tag = ANGLE_TAGS.get(shot.angle, shot.angle)
    if angle_tag: cam_tags.append(angle_tag)
    if shot.movement and shot.movement not in ("static", ""):
        cam_tags.append(shot.movement)
    if shot.lens:
        cam_tags.append(f"{shot.lens} lens")
    if cam_tags:
        parts.append(", ".join(cam_tags))

    # 4. Quality boosters
    parts.append("masterpiece, intricate detail, sharp focus")

    # 5. Scene description from beat — this is the actual visual content
    if beat and beat.description:
        parts.append(beat.description)
    elif scene:
        parts.append(scene.location)

    # 6. Mood / emotion
    if shot.emotion and shot.emotion != "neutral":
        parts.append(f"{shot.emotion} mood")

    # 7. Environment state (time, weather, palette from VisualContinuityStage)
    # (Pushed to the end because if it gets truncated by the 77-token limit,
    # the shot still retains its primary visual and structural anchors).
    if scene and scene.state:
        s = scene.state
        env_parts = []
        if s.time:              env_parts.append(s.time)
        if s.weather:           env_parts.append(s.weather)
        if s.season:            env_parts.append(s.season)
        if s.environment_state: env_parts.append(s.environment_state)
        if env_parts:
            parts.append(", ".join(env_parts))
        if s.lighting:
            parts.append(s.lighting)
        if s.palette:
            parts.append(s.palette)
    else:
        # Fallback: use bible location defaults
        if bible and scene and scene.location in bible.locations:
            loc = bible.locations[scene.location]
            if loc.time_defaults:    parts.append(loc.time_defaults)
            if loc.weather_defaults: parts.append(loc.weather_defaults)

    return ", ".join(p for p in parts if p)


def _build_negative(style_name: str) -> str:
    style = STYLE_TOKENS.get(style_name, STYLE_TOKENS[_DEFAULT_STYLE])
    return style["negative"]


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
        return f"{self.get_name()}_ast_builder_v3.0"

    def execute(self, context: PipelineContext) -> StageResult:
        shot_manifest = None
        scene_manifest = None
        bible = None

        # Prefer the most recent ShotManifest (CastPlanner / CameraPlanner
        # both produce enriched copies — we want the last one).
        for node in reversed(context.execution_nodes):
            if shot_manifest is None and isinstance(node.artifact, ShotManifest):
                shot_manifest = node.artifact
            elif scene_manifest is None and isinstance(node.artifact, SceneManifest):
                scene_manifest = node.artifact
            elif bible is None and isinstance(node.artifact, StoryBible):
                bible = node.artifact

        if not shot_manifest:
            raise ValueError("PromptBuilder requires ShotManifest.")
        if not scene_manifest:
            raise ValueError("PromptBuilder requires SceneManifest.")

        # Build fast lookups
        # scene_hash → Scene
        hash_to_scene = {}
        for s in scene_manifest.scenes:
            h = hashlib.sha256(s.scene_id.encode()).hexdigest()[:8]
            hash_to_scene[h] = s

        # (scene_hash, beat_id) → Beat
        beat_lookup: Dict[tuple, Beat] = {}
        for s in scene_manifest.scenes:
            h = hashlib.sha256(s.scene_id.encode()).hexdigest()[:8]
            for b in s.beats:
                beat_lookup[(h, b.beat_id)] = b

        # Determine project-level director style from any shot
        style_name = _DEFAULT_STYLE
        if shot_manifest.shots:
            raw = shot_manifest.shots[0].style or ""
            style_name = raw.lower() if raw.lower() in STYLE_TOKENS else _DEFAULT_STYLE

        prompts = []
        for shot in shot_manifest.shots:
            parts = shot.shot_id.split("_")
            scene_hash = parts[1] if len(parts) > 1 else ""

            matching_scene = hash_to_scene.get(scene_hash)
            beat = beat_lookup.get((scene_hash, shot.beat_id))

            # --- Assemble the human-readable prompt string ---
            full_prompt = _build_prompt(shot, beat, matching_scene, bible, style_name)
            full_negative = _build_negative(style_name)

            # --- Legacy AST fields (kept for downstream stages that read them) ---
            location = matching_scene.location if matching_scene else "unknown location"
            time_of_day = "day"
            weather = "clear"
            if bible and location in bible.locations:
                loc_profile = bible.locations[location]
                time_of_day = loc_profile.time_defaults or time_of_day
                weather = loc_profile.weather_defaults or weather

            ast_chars = []
            for cm in shot.cast:
                char_ast = CharacterAST(
                    name=cm.character_id,
                    emotion=cm.emotion,
                    pose=cm.pose,
                    visibility=cm.visibility,
                    interaction=cm.interaction,
                )
                if bible and cm.character_id in bible.characters:
                    profile = bible.characters[cm.character_id]
                    char_ast.appearance_tags = [
                        profile.appearance.hair, profile.appearance.eyes,
                        profile.appearance.face, profile.appearance.age,
                        profile.appearance.body,
                    ]
                    char_ast.clothing_tags = [profile.appearance.clothing] + profile.appearance.color_palette
                    char_ast.signature_tags = profile.appearance.signature
                    char_ast.bindings = profile.bindings.model_dump()
                ast_chars.append(char_ast)

            state = matching_scene.state if matching_scene else None
            lighting_style = (state.lighting if state and state.lighting else "cinematic lighting")
            palette = (state.palette if state and state.palette else "")

            seed_hash = hashlib.md5(shot.shot_id.encode()).hexdigest()
            seed = int(seed_hash, 16) % (2**32 - 1)

            ast = PromptAST(
                subject=SubjectAST(description=beat.description if beat else f"{shot.purpose} shot"),
                characters=ast_chars,
                environment=EnvironmentAST(location=location, time_of_day=time_of_day, weather=weather),
                camera=CameraAST(
                    type=shot.camera_type, lens=shot.lens, angle=shot.angle,
                    distance=shot.distance, movement=shot.movement,
                ),
                lighting=LightingAST(style=lighting_style),
                composition=CompositionAST(style="cinematic framing"),
                quality=QualityAST(tags=["masterpiece", "high resolution", "intricate detail"]),
                negative=NegativeAST(tags=full_negative.split(", ")),
            )

            from core.domain.prompt.visual_scene import (
                VisualScene, VisualCharacter, VisualEnvironment, VisualCamera, VisualStyle
            )
            v_chars = []
            for cm in shot.cast:
                profile = bible.characters.get(cm.character_id) if bible else None
                if profile:
                    app = profile.appearance
                    app_desc = (
                        f"Hair: {app.hair}, Eyes: {app.eyes}, Face: {app.face}, "
                        f"Age: {app.age}, Body: {app.body}"
                    )
                    wardrobe_lock = (
                        matching_scene.state.wardrobe_locks.get(cm.character_id, "default")
                        if matching_scene and matching_scene.state else "default"
                    )
                    wardrobe_desc = (
                        ", ".join(bible.wardrobe.get(cm.character_id, {}).get(wardrobe_lock, []))
                        if bible and hasattr(bible, "wardrobe") else ""
                    )
                    bindings_list = [b.model_dump() for b in profile.bindings]
                else:
                    app_desc = "Unknown appearance"
                    wardrobe_desc = "Unknown wardrobe"
                    bindings_list = []

                v_chars.append(VisualCharacter(
                    name=cm.character_id, appearance=app_desc, wardrobe=wardrobe_desc,
                    pose=cm.pose, emotion=cm.emotion, interaction=cm.interaction,
                    visibility=cm.visibility, bindings=bindings_list,
                ))

            v_env = VisualEnvironment(
                location_desc=location,
                time=state.time if state else time_of_day,
                season=state.season if state else "",
                weather=state.weather if state else weather,
                lighting=lighting_style,
                palette=palette,
                environment_state=state.environment_state if state else "",
            )
            v_cam = VisualCamera(
                type=shot.camera_type, lens=shot.lens, angle=shot.angle,
                distance=shot.distance, movement=shot.movement,
            )
            v_style = VisualStyle(
                composition="cinematic framing",
                lighting_style=lighting_style,
                color_grade=palette,
                mood=shot.emotion,
                quality_tags=["masterpiece", "high resolution", "intricate detail"],
                negative_tags=full_negative.split(", "),
            )
            visual_scene = VisualScene(
                subject=beat.description if beat else f"{shot.purpose} shot emphasizing {shot.focus}",
                characters=v_chars, environment=v_env, camera=v_cam, style=v_style,
                props=[], vehicles=[],
            )

            entry = PromptManifestEntry(
                prompt_id=f"prompt_{seed_hash[:8]}",
                scene_id=matching_scene.scene_id if matching_scene else "unknown",
                shot_id=shot.shot_id,
                ast=ast,
                visual_scene=visual_scene,
                seed=seed,
            )
            # Attach pre-assembled strings directly to the AST subject for
            # DiffusersCompiler to consume via full_prompt on LogicalRenderPlan.
            entry.ast.subject.description = full_prompt
            prompts.append(entry)

        manifest = PromptManifest(
            prompts=prompts,
            generator="PromptBuilderStage",
            generator_version="3.1.0",
        )

        node = ExecutionNode(artifact=manifest, stage_name="PromptBuilderStage")
        return StageResult(
            artifact=manifest,
            execution_node=node,
            metrics={"prompts_generated": len(prompts), "style": style_name},
            metadata={},
        )
