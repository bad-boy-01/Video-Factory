import random
import time
import uuid
import logging
from typing import Dict, Any, List
from pathlib import Path

from plugins.interfaces import LLMProvider, ImageGenerationProvider, VideoRendererProvider
from core.pipeline.context import PipelineContext


logger = logging.getLogger(__name__)

class MockLLMProvider(LLMProvider):
    def generate_json(self, prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        logger.info({"provider": "MockLLMProvider", "event": "generate_json", "prompt_len": len(prompt)})
        
        # Look at schema to figure out what to mock
        schema_str = str(schema)
        if "translation" in schema_str:
            # Called by _translate_chinese() in compiler_api.plan()
            return {"translation": "Mock translated text."}
        elif "scenes" in schema_str:
            return {"scenes": [{"scene_id": "mock_scene_1", "characters": ["mock_char_1"], "beats": [{"beat_id": "beat_1", "description": "mock", "emotion": "mock"}]}]}
        elif "beats" in schema_str:
            return {"beats": [{"beat_id": "mock_beat_1", "shots": [
                {"purpose": "mock", "emotion": "mock", "focus": "mock", "importance": "high", "duration": 2.0}
            ]}]}
        elif "cast" in schema_str:
            import re
            shot_ids = re.findall(r'\[Shot (shot_[^_]+_\d+)\]', prompt)
            if not shot_ids:
                shot_ids = ["shot_mock_001"]
            return {"shots": [
                {
                    "shot_id": s_id,
                    "cast": [
                        {
                            "character_id": "mock_char_1",
                            "emotion": "neutral",
                            "pose": "standing",
                            "visibility": "foreground",
                            "interaction": "none"
                        }
                    ]
                } for s_id in shot_ids
            ]}

        elif "prompts" in schema_str:
            return {"prompts": [
                {
                    "shot_id": "shot_mock_001",
                    "ast": {
                        "subject": {"description": "mock"},
                        "camera": {"distance": "mock"},
                        "mood": {"mood": "mock"},
                        "technical": {"width": 1024, "height": 1024, "steps": 1, "cfg": 7.0}
                    }
                }
            ]}
            
        return {"characters": [{"name": "mock", "visual_dna": "mock", "outfit": "mock", "color_palette": "mock"}]}

    def initialize(self) -> None:
        pass

    def load(self) -> None:
        logger.info({"provider": "MockLLMProvider", "event": "load"})

    def unload(self) -> None:
        logger.info({"provider": "MockLLMProvider", "event": "unload"})

    def shutdown(self) -> None:
        pass

    def generate_text(self, prompt: str, system_prompt: str = "") -> str:
        return "Mock response text."


class MockImageGenerator(ImageGenerationProvider):
    def get_model_name(self) -> str:
        return "mock-diffusion-v1"

    def get_model_revision(self) -> str:
        return "1.0.0-mock"

    def generate_image(self, prompt: str, negative_prompt: str, seed: int, output_path: Path) -> Path:
        logger.info({"provider": "MockImageGenerator", "event": "generate_image", "seed": seed, "output": str(output_path)})
        time.sleep(0.1) # Simulate some work
        
        # Simulate occasional corruption
        if random.random() < 0.05:
            logger.error({"provider": "MockImageGenerator", "event": "corruption", "detail": "Simulated image generation failure"})
            raise RuntimeError("CUDA out of memory (simulated)")

        # Create a dummy image file with varying size
        size_bytes = random.randint(50000, 150000)
        with open(output_path, 'wb') as f:
            f.write(os.urandom(100) if random.random() < 0.01 else b'0' * size_bytes) # Occasional invalid small file
            
        return output_path


# class MockEvaluator(EvaluatorPlugin):
#     def __init__(self, name: str = "MockConsistencyEvaluator"):
#         self.name = name
# 
#     def get_name(self) -> str:
#         return self.name
# 
#     def evaluate(self, asset: Asset, context: PipelineContext) -> EvaluationResult:
#         logger.info({"provider": "MockEvaluator", "event": "evaluate", "asset": str(asset.file_path)})
#         
#         # Check for simulated corruption
#         if asset.file_path.exists() and asset.file_path.stat().st_size < 1000:
#             return EvaluationResult(score=0.1, reason="Image file is corrupted or too small", retry_needed=True)
# 
#         # Vary score to simulate borderline cases
#         score = random.uniform(0.65, 1.0)
#         
#         retry = False
#         reason = "Pass"
#         if score < 0.8:
#             retry = True
#             reason = "Identity drift detected (simulated)"
#             logger.warning({"provider": "MockEvaluator", "event": "failure", "score": score, "reason": reason})
# 
#         return EvaluationResult(score=score, reason=reason, retry_needed=retry)


class MockVideoRenderer(VideoRendererProvider):
    def render_video(self, manifest: Any, audio_paths: List[Path], output_path: Path) -> Path:
        logger.info({"provider": "MockVideoRenderer", "event": "render_video", "frames": len(manifest.frames) if hasattr(manifest, 'frames') else 0})
        time.sleep(0.2) # Simulate encoding delay
        
        if random.random() < 0.02:
            raise RuntimeError("FFmpeg encoding failed (simulated)")

        with open(output_path, 'wb') as f:
            f.write(b'fake_video_data')
            
        return output_path

import os
