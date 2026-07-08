"""
CharacterSheetsStage
====================
Generates 7-pose reference images for every character in story_bible.json.

Poses: front | side | three_quarter | smiling | angry | sad | action

Storage:  workspace/characters/<character_name>/<pose>.png
  The three_quarter.png is wired as the IP-Adapter reference in render().

Idempotency: Skips a character if all 7 pose files already exist.
  Pass force=True to regenerate regardless.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

logger = logging.getLogger(__name__)

POSE_CONFIGS = [
    {"pose": "front",         "description": "full-body front view, neutral expression, arms at sides", "framing": "full body shot",  "mood": "neutral"},
    {"pose": "side",          "description": "full-body side profile, neutral expression",              "framing": "full body shot",  "mood": "neutral"},
    {"pose": "three_quarter", "description": "three-quarter view, slight turn, natural expression",    "framing": "medium shot",     "mood": "calm"},
    {"pose": "smiling",       "description": "portrait, warm smile, looking at camera",                "framing": "close-up shot",   "mood": "happy"},
    {"pose": "angry",         "description": "portrait, furrowed brow, intense expression",            "framing": "close-up shot",   "mood": "angry"},
    {"pose": "sad",           "description": "portrait, downcast eyes, melancholy expression",         "framing": "close-up shot",   "mood": "sad"},
    {"pose": "action",        "description": "dynamic action pose, full body, motion implied",         "framing": "wide shot",       "mood": "intense"},
]

ALL_POSES = {cfg["pose"] for cfg in POSE_CONFIGS}


class CharacterSheetsStage:
    """Generate multi-angle reference images for all characters in the story bible."""

    def __init__(self, story_bible_path, characters_root, provider, compiler, force=False):
        self.story_bible_path = story_bible_path
        self.characters_root = characters_root
        self.provider = provider
        self.compiler = compiler
        self.force = force

    def run(self):
        """Run all character sheet generation. Returns dict: generated, skipped, failed."""
        characters = self._load_characters()
        if not characters:
            logger.warning("CharacterSheetsStage: no characters found in story_bible.json.")
            return {"generated": 0, "skipped": 0, "failed": 0}

        generated = skipped = failed = 0

        for char_id, char_data in characters.items():
            char_name = char_data.get("name", char_id).lower().replace(" ", "_")
            char_dir = self.characters_root / char_name
            char_dir.mkdir(parents=True, exist_ok=True)

            if not self.force and self._all_poses_exist(char_dir):
                logger.info(f"[CharSheets] Skipping '{char_name}' - all 7 poses already exist.")
                skipped += 1
                continue

            logger.info(f"[CharSheets] Generating reference sheet for '{char_name}'...")
            g, f = self._generate_character(char_name, char_data, char_dir)
            generated += g
            failed += f

        return {"generated": generated, "skipped": skipped, "failed": failed}

    def _load_characters(self):
        try:
            data = json.loads(self.story_bible_path.read_text(encoding="utf-8"))
            return data.get("characters", {})
        except Exception as e:
            logger.error(f"[CharSheets] Failed to read story_bible.json: {e}")
            return {}

    def _all_poses_exist(self, char_dir):
        return all((char_dir / f"{pose}.png").exists() for pose in ALL_POSES)

    def _build_appearance_text(self, char_data):
        """Flatten the Appearance dict into a comma-separated prompt fragment."""
        appearance = char_data.get("appearance", {})
        if isinstance(appearance, str):
            return appearance

        parts = []
        for field in ("hair", "eyes", "face", "age", "body", "clothing"):
            val = appearance.get(field, "")
            if val:
                parts.append(val)

        color_palette = appearance.get("color_palette", [])
        if isinstance(color_palette, list) and color_palette:
            parts.append("color palette: " + ", ".join(color_palette))

        signature = appearance.get("signature", [])
        if isinstance(signature, list) and signature:
            parts.append("signature features: " + ", ".join(signature))

        return ", ".join(parts) if parts else "generic character"

    def _generate_character(self, char_name, char_data, char_dir):
        """Generate all 7 pose images for a single character. Returns (generated, failed)."""
        appearance_text = self._build_appearance_text(char_data)
        name_display = char_data.get("name", char_name)

        generated = failed = 0

        for pose_cfg in POSE_CONFIGS:
            pose = pose_cfg["pose"]
            output_path = char_dir / f"{pose}.png"

            if not self.force and output_path.exists():
                logger.info(f"  [CharSheets] '{char_name}/{pose}' already exists, skipping.")
                generated += 1
                continue

            subject = f"{name_display}, {appearance_text}, {pose_cfg['description']}"
            prompt_text = (
                f"({subject}:1.3), "
                f"({pose_cfg['framing']}:1.1), "
                f"({pose_cfg['mood']}:1.0), "
                f"cinematic lighting, high detail, character reference sheet"
            )
            negative_text = (
                "low quality, blurry, distorted, bad anatomy, watermark, "
                "multiple people, crowd, background characters"
            )

            try:
                from core.domain.prompt.render_plan import RenderPlan, LogicalRenderPlan, PhysicalRenderPlan

                plan = RenderPlan(
                    shot_id=f"charsheet_{char_name}_{pose}",
                    logical=LogicalRenderPlan(
                        subject=subject,
                        framing=pose_cfg["framing"],
                        emphasis="character reference sheet",
                        mood=pose_cfg["mood"],
                    ),
                    physical=PhysicalRenderPlan(
                        width=768,
                        height=1024,
                        steps=25,
                        cfg=7.5,
                        seed=abs(hash(f"{char_name}_{pose}")) % (2 ** 31),
                    ),
                )
                request = self.compiler.compile_plan(plan)

                # Override prompt with appearance-specific text
                request.conditioning.prompt = prompt_text
                request.conditioning.negative_prompt = negative_text

                image = self.provider.generate(request)
                image.save(output_path)
                logger.info(f"  [CharSheets] Generated '{char_name}/{pose}' -> {output_path}")
                generated += 1

            except Exception as e:
                logger.error(f"  [CharSheets] Failed '{char_name}/{pose}': {e}")
                failed += 1

        return generated, failed
