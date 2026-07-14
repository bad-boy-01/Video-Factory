# Agent Handoff & Context Document

**Instructions for future AI Agents**: 
When the user attaches new Kaggle logs or asks you to fix an issue, **READ THIS DOCUMENT FIRST**. It contains critical context about the architecture, recent changes, project rules, and environment constraints.

---

## 1. Project Rules & Guidelines
The following rules apply specifically to the NovelFactory project workspace and must be followed strictly by all agents:

- **Kaggle Environment Optimization**: When implementing model downloading or caching logic (e.g., HuggingFace pipelines, Diffusers), **always** override the default `cache_dir` to `/tmp/models`. This prevents the `/kaggle/working/` directory from bloating and exceeding disk space limits during execution.
- **Deterministic Pipeline Architecture**: NovelFactory is a deterministic AI video compiler, not a generic text-to-video API wrapper. Never replace structured JSON schema stages (e.g., `SceneSplitterStage`, `ShotPlannerStage`) with free-form LLM prose generation. To integrate new cinematic styles or features (like ViMax styles), extend the JSON schemas with explicit metadata fields (like `scene_style` and `duration` limits) instead of relying on the LLM to output unconstrained paragraphs.
- **Agent Handoff Protocol**: At the start of every session, read the `AGENT_HANDOFF.md` file in its entirety to understand the current state of the project. At the end of every session, append new logs and architectural updates to `AGENT_HANDOFF.md`. This guarantees context continuity across agent sessions, as the user frequently clears previous chat histories.

---

## 2. How NovelFactory Works: The AI Film Compiler
NovelFactory is an end-to-end "AI Film Compiler" that converts raw unstructured text (such as a novel or a screenplay) into a fully animated, temporally coherent video. Unlike generic text-to-video APIs (like Sora or Runway) where you type a paragraph and hope for a good video, NovelFactory acts as a **deterministic operating system**. It breaks down the generation process into strict, mathematically controlled stages to guarantee character consistency and narrative accuracy.

Here is the complete step-by-step breakdown of how the engine works:

### Phase A: Creative Planning (The Compiler Frontend)
In this phase, we use Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) to understand the story and build a blueprint for the video.
1. **ChunkerStage**: A typical novel has 100,000+ words. An LLM cannot read this all at once. The Chunker breaks the text into manageable blocks (chunks) while maintaining overlaps so context is never lost.
2. **StoryBibleGeneratorStage**: Reads the chunks and extracts "Global Constants": characters, locations, and lore. It creates a definitive "Story Bible" so the AI remembers that the protagonist has "blue eyes and a scar" throughout the entire film.
3. **SceneSplitterStage**: Slices the text chunks into exact narrative "Scenes". It keeps mathematical track of character offsets so not a single sentence of the original book is skipped.
4. **ShotPlannerStage (with ViMax Cinematic Logic)**: Converts narrative beats into a cinematic shot list. It acts as the "Director". It automatically classifies the scene style. If the scene is an **Action/Motion** scene, it forces rapid cuts (1.0 - 2.0s duration). If it's a **Narrative** scene, it uses standard coverage (Establishing -> Over-the-shoulder -> Close-up) with longer durations.
5. **CastPlanner & CameraPlanner**: Assign the specific global characters to the shots, and calculate virtual camera movements (pans, tilts, zooms).
6. **PromptBuilderStage**: Translates all this JSON metadata into highly optimized comma-separated tags (positive and negative prompts) that diffusion models natively understand.

### Phase B: Rendering (The GPU Backend)
In this phase, we use local Stable Diffusion pipelines to generate the actual pixels. 
1. **Local Diffusion (SDXL Lightning)**: We generate high-quality still images using `ByteDance/SDXL-Lightning` instead of direct video generation, prioritizing aesthetic quality and prompt adherence over temporal flickering.
2. **Character Consistency (IP-Adapter & Memory)**: To prevent characters from shape-shifting between shots, we inject an **IP-Adapter** guided by a `reference_sheet.png`. The `CharacterVisualProfile` explicitly tracks continuity fields (`face_seed`, `wardrobe_locks`, `current_injuries`) which are appended to the prompt by the `PromptCompiler`.
3. **ArtifactStore (CAS) & Incremental DAG**: Generated images are stored in a Content-Addressable Storage (CAS) system keyed by prompt hash. The `ArtifactDAG` monitors dependency hashes to skip upstream stages that haven't changed, offering extreme efficiency and chapter-level resumability.
4. **ImageQA & Critic Loop**: Generated images undergo multi-metric evaluation (CLIP adherence, sharpness, OCR artifact detection). Failing images trigger a deterministic Critic feedback loop to revise the prompt and regenerate.

### Phase C: Assembly (The Linker)
The final phase stitches all the disparate assets into a seamless movie.
1. **Audio Generation**: We generate dialogue and sound effects (using TTS systems like Kokoro) to match the pacing dictated by the ShotPlanner.
2. **FFmpeg Renderer**: A deterministic script takes the 100+ generated `.mp4` video clips, applies crossfades, overlays subtitles (typography), syncs the audio tracks, and outputs the final Master Video File. 

---

## 3. Next Steps for the AI
1. Read the newly attached `kaggle_log.txt` or whatever logs the user provides.
2. Identify the failure point (e.g., VRAM Out of Memory, FFmpeg stitch failure, AnimateDiff tensor mismatch).
3. Apply fixes adhering to the deterministic architecture described above. Do **NOT** revert to static image generation unless explicitly instructed.
4. **Append** a summary of any new architectural decisions you make to the end of this document.

---

## Agent Updates

**Date**: 2026-07-09
**Issue Fixed**: AnimateDiff crashed with `ValueError: Incompatible Motion Adapter, got different number of blocks` during the rendering phase.
**Resolution/Decision**: The `diffusion_model` / `model_id` configuration defaulted to `stabilityai/stable-diffusion-xl-base-1.0`. Since the project architecture is strictly locked to Stable Diffusion 1.5 for the AnimateDiff video backend (`animatediff-motion-adapter-v1-5-2`), loading an SDXL base model caused a tensor shape mismatch. The default model has been updated to `runwayml/stable-diffusion-v1-5` across `DiffusionConfig`, `ModelConfig`, and `RenderingConfig` to restore compatibility and comply with the VRAM/architecture requirements.

**Date**: 2026-07-09
**Issue Fixed**: character_sheets() failed ('NoneType' object has no attribute 'tokenize') and IP-Adapter expected shape [320, 768] but got [640, 2048] during the rendering phase.
**Resolution/Decision**: The previous fix changed the base model to Stable Diffusion 1.5, but DiffusersProvider (used by character_sheets) was hardcoded to use StableDiffusionXLPipeline and load SDXL's IP-Adapter weights. Updated DiffusersProvider in plugins/local_diffusion.py to dynamically select StableDiffusionPipeline or StableDiffusionXLPipeline, and the corresponding IP-Adapter bin (ip-adapter_sd15.bin vs ip-adapter_sdxl.bin), based on whether 'sdxl' is in the model_id.

**Date**: 2026-07-11
**Issue Fixed**: Stable Diffusion 1.5 Token Limit Truncation (Token indices sequence length > 77) during rendering phase.
**Resolution/Decision**: `PromptBuilderStage` previously appended long scene state and environment descriptions before critical character and camera details, causing the most important visual anchors to be discarded by the 77-token CLIP limit. Reordered the string assembly in `core/optimization/prompt_builder.py` to prioritize style, characters, and camera constraints before environmental and beat description fields.

**Date**: 2026-07-14
**Major Architecture Overhaul**: The AnimateDiff pipeline was deemed too unstable and low-quality under constrained (free-tier) limits. The system was transformed into an AI-Directed Film Compiler for Korean manhwa-style videos.
**Resolutions/Decisions**:
1. **Rendering Backend Swap**: Replaced AnimateDiff with SDXL Lightning (`ByteDance/SDXL-Lightning`) generating still images.
2. **Direction Layer**: Introduced `NarrativeAnalyzer`, `StoryboardPlanner`, `CinematographyEngine` and `SceneGraphBuilder`. The LLM now focuses on facts (genre, emotion, entity mapping), while deterministic policy engines handle camera decisions and composition.
3. **ArtifactStore & Image Bank**: Replaced `AssetRegistry` with a content-addressable `ArtifactStore`. Generated images are stored by `prompt_hash`, allowing instant reuse across shots with identical prompts, drastically saving Kaggle GPU time.
5. **Post-Production (FFmpeg)**: Upgraded `FFmpegVideoRenderer` to perform Ken Burns motion (zoompan) on the SDXL stills, crossfade transitions (xfade), and SRT subtitle burn-in.
6. **Character Memory**: Upgraded `CharacterVisualProfile` to include strict continuity fields (`face_seed`, `current_outfit`, `current_injuries`, etc.) to enforce visual consistency across prompts.
7. **Incremental Rebuilds (DAG)**: Built `ArtifactDAG` and wired it into `CompilerExecutor` to compute logical dependency hashes, safely skipping unmodified upstream pipeline stages.
