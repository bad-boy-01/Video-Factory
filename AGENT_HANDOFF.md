# Agent Handoff & Context Document

**Instructions for future AI Agents**: 
When the user attaches new Kaggle logs or asks you to fix an issue, **READ THIS DOCUMENT FIRST**. It contains critical context about the architecture, recent changes, and environment constraints.

## 1. What We Are Building
We are building **NovelFactory**, a deterministic AI Film Compiler. 
As described in the `README.md`, it is a "Generative Operating System" designed to convert unstructured text (novels/scripts) into temporally and visually coherent rendered videos. It relies on strict separation between:
1. **Creative Planning**: Parsing text into scenes, characters, and shots.
2. **GPU Orchestration**: Rendering frames (or video clips) deterministically.
3. **Assembly**: Using FFmpeg to compile the final video with audio.

## 2. Recent Major Architectural Pivot (Local Text-To-Video)
- **Goal**: The user wants animated video clips generated *directly* from the script while maintaining character consistency, rather than a slideshow of static images.
- **Implementation**: We implemented **Option 2** (Local Animation) by introducing `AnimateDiffProvider` in `plugins/local_diffusion.py`. 
- **Tech Stack**: It uses `AnimateDiffPipeline` with the `guoyww/animatediff-motion-adapter-v1-5-2` adapter (Stable Diffusion 1.5).
- **Character Consistency**: We still load the `IP-Adapter` (using `ip-adapter_sd15.bin`) and feed it the `reference_sheet.png` generated during the planning stage. This forces the AnimateDiff video output to maintain the same characters across different video clips.
- **Assembly**: `core/api/compiler_api.py` and `plugins/ffmpeg_renderer.py` have been updated to output `.mp4` video clips (16 frames each) instead of `.png` images, and stitch these video clips seamlessly using FFmpeg concat.

## 3. Kaggle Environment Constraints & Fixes
- **VRAM Limitations**: Kaggle (T4 x2) has 16GB VRAM per GPU. SDXL AnimateDiff is too heavy, which is why the video generation stage explicitly falls back to SD1.5. 
- **Storage / Disk Space Bloat**: Previously, downloading massive model weights (SD1.5, AnimateDiff, IP-Adapter, Qwen LLM) to `/kaggle/working/workspace/models` resulted in a 6.6GB Kaggle output zip file (since Kaggle automatically zips everything in `/kaggle/working` at the end of a run). 
- **The Fix**: The `cache_dir` for diffusion models and presets has been permanently changed to `/tmp/models/` in `plugins/interfaces.py`, `core/domain/pipeline_config.py`, and `core/domain/rendering/presets.py`. `/tmp/` is ignored by Kaggle's output zipping mechanism.

## 4. ViMax Context
- The repository contains a `ViMax-main` module. ViMax was evaluated earlier, but it relies heavily on external Cloud APIs (Doubao/Omni-Flash) for video generation. 
- The user explicitly requested to proceed with the **local, on-device animation approach (AnimateDiff)** to ensure we can strictly enforce character consistency via IP-Adapter, which is difficult to guarantee with external APIs.

## 5. Next Steps for the AI
1. Read the newly attached `kaggle_log.txt` or whatever logs the user provides.
2. Identify the failure point (e.g., VRAM Out of Memory, FFmpeg stitch failure, AnimateDiff tensor mismatch).
3. Apply fixes adhering to the architecture described above. Do **NOT** revert to static image generation unless explicitly instructed by the user.
