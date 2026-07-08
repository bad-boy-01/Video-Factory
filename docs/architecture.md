# NovelFactory Architecture

NovelFactory is an AI film compiler. It transforms narrative text into deterministic RenderPlans and executes them via pluggable video/image generation providers.

## The Compiler Flow
1. **Semantic Planning**: StoryBible -> SceneManifest -> BeatGraph -> SceneGraph -> RenderPlan
2. **Rendering Execution**: RenderPlan -> ProviderRequest -> RenderScheduler -> RenderGraph -> Image/Video
3. **Storage**: All assets are managed in a Content Addressable Storage (CAS) registry.

## Guiding Principles
- **Frozen Architecture**: No new model or capability should require redesigning the upstream semantic IRs.
- **Reproducibility**: Global seeds, locked schemas, and deterministic state tracking guarantee repeatability.
