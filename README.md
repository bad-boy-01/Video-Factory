# NovelFactory: AI Film Compiler

NovelFactory is a **deterministic, resumable, artifact-driven AI video compiler**. It is designed to solve the fundamental problem of converting unstructured text (like novels or scripts) into temporally and visually coherent rendered videos ("visual novels") by strictly separating creative planning, GPU orchestration, and closed-loop QA.

---

## 1. Core Philosophy

In traditional generative systems, pipelines assume *same input → same output*. However, under real LLM and Diffusion variance, this breaks down. NovelFactory shifts the paradigm from simple generation scripts to a **Generative Operating System**:

* **Immutable Artifacts**: Every step of the pipeline produces an immutable, hash-verified artifact (e.g., `SceneManifest`, `PromptManifest`, `Timeline`). 
* **Resumable Execution**: A SQLite-backed `RenderQueue` ensures that if a Kaggle kernel crashes on shot 45 out of 120, it safely resumes exactly where it left off.
* **Separation of State**: Canonical data (Story Bible) is strictly separated from ephemeral render data (`CharacterRenderState`), keeping identity tracking pure.

---

## 2. Architecture: The AI Film Compiler

The system breaks down narrative texts exactly like a compiler processes source code:

1. **Creative Planning (IR Construction)**
   - `ChunkerStage`: Safely segments 500k-word novels into manageable contexts with overlapping boundaries.
   - `SceneSplitterStage`: Emits a `SceneManifest` featuring stable, deterministic scene IDs.
   - `StoryContinuityEngineStage`: Generates a `SceneMemory` object to track temporal weather, lighting, inventory, and character states across scenes.

2. **Optimization (AST Refinement)**
   - `PromptAST`: Highly modular representations of Shots (`Subject`, `Environment`, `Camera`, `Lighting`).
   - `PromptOptimizerStage`: A pass that deterministically optimizes the generic AST into model-specific syntax (e.g., SDXL vs Flux).
   - `PromptValidatorStage`: A static analysis check that traps empty, invalid, or banned prompts before touching the GPU.

3. **Audio & NLE Timeline (Linking)**
   - `AudioGenerationStage`: Integrates TTS (like Kokoro TTS) to drive exact shot durations (narrative pacing).
   - `TimelineBuilderStage`: Emits a Premiere Pro-style `Timeline` featuring prioritized `VideoTrack`, `SubtitleTrack`, and `VoiceTrack`s containing explicit mathematical animations (Ken Burns).

4. **Rendering & Closed-Loop QA (Execution)**
   - `ImageGenerationProvider`: Interfaces to swap between `SDXL`, `SD1.5`, or future architectures.
   - `AssetRegistry`: Tracks dependencies. If a LoRA or prompt changes, the exact cached image is invalidated.
   - `ImageQAStage`: A dual-pass (Fast QA + Semantic Vision LLM) system that intercepts failed renders and pushes them back into the `RenderQueue` as `REPAIR_PENDING`.

5. **Assembly**
   - `FFmpegAssemblyStage`: "Dumb" execution of the layered Timeline, rendering crossfades, subtitle typography, and audio mixes.

---

## 3. Directory Structure

The codebase is organized by strict responsibilities:

```text
core/
├── domain/       # Immutable domain objects (ProjectManifest, AssetRegistry, Timeline)
├── planning/     # Creative LLM stages (Chunker, Scene Splitter, Planners)
├── optimization/ # AST conversion and formatting
├── validation/   # Prompt static analysis & Closed-loop Image QA
└── rendering/    # GPU Orchestration (RenderQueue, Image Providers, Audio, FFmpeg)
```

---

## 4. Usage & Execution Harness

### Installation

```bash
git clone https://github.com/bad-boy-01/NovelFactory.git
cd NovelFactory
pip install -r requirements.txt
```

### Running the Orchestrator

NovelFactory utilizes a sequential executor to pipe the `PipelineContext` through the compiler stages.

```bash
python main.py --novel my_script.txt --stage all
```

**What happens?**
* The system builds a unified `ProjectManifest`.
* It expands the novel, builds the Timeline, and calculates the exact frame durations.
* The `RenderQueue` executes diffusion jobs, tracking progress in SQLite.
* Failed images are caught by the `ImageQAStage` and automatically repaired.
* `FFmpeg` stitches the final `Timeline` into a cinematic `.mp4`.
