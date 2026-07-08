# NovelFactory Capability Roadmap (Milestones 10–15)

With the compiler architecture formally frozen in Milestone 9, NovelFactory transitions entirely from building **infrastructure** to delivering **capabilities**. No further changes will be made to upstream planning stages (`StoryBible` -> `RenderPlan`) to support new models. Future work is strictly focused on improving visual quality, generation speed, and model support through plugins.

## The Ultimate Vision
Given a long-form novel, NovelFactory automatically compiles a 2–3 hour cinematic video with **zero manual editing**, designed specifically for headless execution on Kaggle's free GPU tier.

### Target Aesthetic: Korean Manhwa / Webtoon
Unlike typical AI video generators aiming for realism or Pixar styles, NovelFactory targets the high-polish, motion-comic aesthetic of Korean webtoons (e.g., *Solo Leveling*, *Omniscient Reader*). Every frame should feel like a perfectly composed manhwa panel.

### The Great Challenge: Continuity at Scale
Generating a 3-hour slideshow requires 1,500–3,000 images. The primary technical hurdle is preventing drift—keeping faces, clothing, lighting, and environments consistent across thousands of discrete generations. Our compiler architecture (`StoryBible`, `SceneGraph`, `RenderPlan`) exists entirely to fight this drift through deterministic planning.

## Milestone 10: Model Ecosystem
**Goal:** Expand provider plugins to support the state-of-the-art open-weight landscape.
- **Image Models:** SDXL Turbo, FLUX Dev/Schnell, SD3, PixArt Sigma.
- **Integration:** All models must implement the existing `ProviderRequest` interface without requiring new IR abstractions.

## Milestone 11: Image Quality (Automated Evaluation)
**Goal:** Guarantee baseline cinematic quality through autonomous retry loops.
- **Implementation:** Introduce an `EvaluateNode` into the `RenderGraph`.
- **Metrics:** Automatically score and reject images based on CLIP similarity, prompt adherence, face quality, anatomy, blur, and text artifacts. Bad generations automatically trigger a seed/parameter retry.

## Milestone 12: Storytelling & Cinematic Continuity
**Goal:** Shift focus from "generating single images" to "directing a film."
- Improve pacing, emotional rhythm, and establishing shots.
- Implement recurring visual motifs and strict character continuity tracking across the `SceneGraph`.

## Milestone 13: Motion & Video Foundation Models
**Goal:** Direct integration of temporally consistent video generation.
- **Video Models:** Wan 2.2, CogVideoX, Hunyuan Video, LTX, Cosmos.
- **Approach:** Map the `ProviderRequest` directly to video generation backends.

## Milestone 14: Audio & Sound Design
**Goal:** Add aural depth to the generated video.
- Automated Music and SFX generation based on `BeatManifest` emotional tagging.
- Dialogue generation and localized lip-sync capabilities.

## Milestone 15: Automated Editing
**Goal:** Post-production automation.
- Automatic cuts, crossfades, camera shake, speed ramps, Ken Burns zoom effects, subtitle generation, and music synchronization.

> [!NOTE]
> The overriding directive for these milestones is **Stability over Abstraction**. The current architecture is rich enough; every new capability must leverage the existing `ProviderRequest`, `RenderGraph`, or CAS systems without introducing new foundational complexity.
