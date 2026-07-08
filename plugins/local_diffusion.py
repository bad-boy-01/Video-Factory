from PIL import Image
from plugins.interfaces import DiffusionConfig, ImageGenerationProvider, ProviderHealth
import logging
import gc

logger = logging.getLogger(__name__)

from core.domain.prompt.render_plan import RenderPlan
from core.domain.prompt.provider_request import ProviderRequest
from plugins.interfaces import ProviderCompiler

class MockCompiler(ProviderCompiler):
    def compile_plan(self, plan: RenderPlan) -> ProviderRequest:
        from core.domain.prompt.provider_request import GenerationParams, ConditioningParams
        return ProviderRequest(
            request_type="image",
            generation=GenerationParams(
                resolution=(plan.physical.width, plan.physical.height),
                steps=plan.physical.steps,
                cfg=plan.physical.cfg,
                seed=plan.physical.seed
            ),
            conditioning=ConditioningParams(
                prompt="Mock Image based on plan",
                negative_prompt=""
            )
        )

class MockProvider(ImageGenerationProvider):
    def __init__(self):
        self.loaded = False
        
    def load(self) -> None:
        pass
        
    def generate(self, request: ProviderRequest, callback=None) -> Image.Image:
        prompt = request.conditioning.prompt if request.conditioning else "Mock"
        steps = request.generation.steps if request.generation else 10
        width = request.generation.resolution[0] if request.generation else 1024
        height = request.generation.resolution[1] if request.generation else 1024
        
        logger.info(f"Mock Generating: {prompt}...")
        if callback:
            for i in range(steps):
                callback(i, steps)
        return Image.new('RGB', (width, height), color='green')

    def generate_batch(self, requests: list, callback=None) -> list:
        if not requests:
            return []
        steps = requests[0].generation.steps if requests[0].generation else 10
        logger.info(f"Mock Generating batch: {len(requests)} shots...")
        if callback:
            for i in range(steps):
                callback(i, steps)
        images = []
        for r in requests:
            width = r.generation.resolution[0] if r.generation else 1024
            height = r.generation.resolution[1] if r.generation else 1024
            images.append(Image.new('RGB', (width, height), color='green'))
        return images
        
    def health_check(self) -> ProviderHealth:
        return ProviderHealth(loaded=True, device="cpu", model="mock", dtype="none", vram_allocated_gb=0.0)
        
    def unload(self) -> None:
        pass

class DiffusersCompiler(ProviderCompiler):
    def __init__(self, config: DiffusionConfig = None):
        self.config = config

    def compile_plan(self, plan: RenderPlan) -> ProviderRequest:
        # Build SDXL Prompt from LogicalRenderPlan
        logical = plan.logical
        
        # 1. Subject (Highest Weight)
        sections = []
        if logical.subject:
            sections.append(f"({logical.subject}:1.2)")
            
        # 2. Framing & Atmosphere
        if logical.framing:
            sections.append(f"({logical.framing}:1.1)")
        if logical.mood:
            sections.append(f"({logical.mood}:1.0)")
        if logical.emphasis:
            sections.append(f"({logical.emphasis}:0.9)")
            
        prompt_str = " ".join(sections)
        negative_str = "low quality, blurry, distorted, bad anatomy, watermark"
        
        from core.domain.prompt.provider_request import GenerationParams, ConditioningParams, BindingParams
        return ProviderRequest(
            request_type="image",
            generation=GenerationParams(
                resolution=(plan.physical.width, plan.physical.height),
                steps=plan.physical.steps,
                cfg=0.0 if self.config and self.config.adapter and "Lightning" in self.config.adapter else plan.physical.cfg,
                seed=plan.physical.seed
            ),
            conditioning=ConditioningParams(
                prompt=prompt_str,
                negative_prompt=negative_str,
                controlnets=plan.physical.controlnets
            ),
            bindings=BindingParams(
                loras=plan.physical.loras
            )
        )

class DiffusersProvider(ImageGenerationProvider):
    def __init__(self, config: DiffusionConfig = None):
        # We assume config is actually a ModelConfig in the new architecture
        import torch
        self.config = config
        self.pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._ip_adapter_loaded = False
        self._faceid_loaded = False
        self._face_app = None
        
    def generate(self, request: ProviderRequest, callback=None) -> Image.Image:
        if not self.pipeline:
            self.load()
            
        import torch
        generator = torch.Generator(device=self.device).manual_seed(request.generation.seed)
        
        logger.info(f"[Inference] Running generation for seed {request.generation.seed} with {request.generation.steps} steps.")
        
        def step_callback(step: int, timestep: int, latents: torch.Tensor):
            if callback:
                callback(step, request.generation.steps)

        kwargs = dict(
            prompt=request.conditioning.prompt,
            negative_prompt=request.conditioning.negative_prompt,
            num_inference_steps=request.generation.steps,
            guidance_scale=request.generation.cfg,
            generator=generator,
            width=request.generation.resolution[0],
            height=request.generation.resolution[1],
            callback=step_callback,
            callback_steps=1
        )
        kwargs.update(self._build_conditioning(request.conditioning.ip_adapter))

        image = self.pipeline(**kwargs).images[0]
        
        return image

    def generate_batch(self, requests: list, callback=None) -> list:
        """
        Generate multiple shots in a single pipeline call. All requests in a
        batch must share the same resolution/steps/cfg (a diffusers batch is
        one forward pass over a single tensor shape) - the caller (render())
        is responsible for only grouping compatible requests together.

        A single ip_adapter_image is used for the whole batch (diffusers
        broadcasts one reference image across every item in a batched call),
        so this is only meaningful when every request in the batch shares the
        same set of characters/references - again, the caller's job to ensure.
        """
        if not requests:
            return []
        if not self.pipeline:
            self.load()

        import torch

        first = requests[0].generation
        generators = [
            torch.Generator(device=self.device).manual_seed(r.generation.seed)
            for r in requests
        ]

        def step_callback(step: int, timestep: int, latents: torch.Tensor):
            if callback:
                callback(step, first.steps)

        # Shared conditioning: use the first request's references - the
        # caller only batches requests that already share the same reference set.
        kwargs = dict(
            prompt=[r.conditioning.prompt for r in requests],
            negative_prompt=[r.conditioning.negative_prompt for r in requests],
            num_inference_steps=first.steps,
            guidance_scale=first.cfg,
            generator=generators,
            width=first.resolution[0],
            height=first.resolution[1],
            callback=step_callback,
            callback_steps=1,
        )
        kwargs.update(self._build_conditioning(requests[0].conditioning.ip_adapter))

        logger.info(f"[Inference] Batched generation: {len(requests)} shots in one call.")
        result = self.pipeline(**kwargs)
        return list(result.images)

    def _build_ip_adapter_image(self, ip_adapter_refs: dict):
        """
        ip_adapter_refs: {character_name: reference_image_path}. Loads and,
        if there's more than one character in the shot, averages them into a
        single composite (same approach as the character-reference manager
        elsewhere in this project) since a single SDXL IP-Adapter slot takes
        one conditioning image, not a per-character list.
        Returns a PIL.Image, or None if there are no valid references.
        """
        if not ip_adapter_refs:
            return None
        try:
            import numpy as np
            paths = [p for p in ip_adapter_refs.values() if p]
            imgs = []
            for p in paths:
                try:
                    imgs.append(Image.open(p).convert("RGB").resize((256, 256), Image.LANCZOS))
                except Exception as e:
                    logger.warning(f"[IP-Adapter] Could not load reference {p}: {e}")
            if not imgs:
                return None
            if len(imgs) == 1:
                return imgs[0]
            avg = np.mean([np.array(i, dtype=np.float32) for i in imgs], axis=0).astype(np.uint8)
            return Image.fromarray(avg)
        except Exception as e:
            logger.warning(f"[IP-Adapter] Failed to build reference composite: {e}")
            return None

    def _build_conditioning(self, ip_adapter_refs: dict) -> dict:
        """
        Decides between Face-ID (embedding-based, single-character, generally
        stronger facial identity preservation) and standard IP-Adapter
        (composite reference image, any number of characters) conditioning
        for one shot, and returns the kwargs to merge into the pipeline call.

        Face-ID is the least-verified part of this provider - it depends on
        insightface (a separate face-analysis library with its own model
        download) and is opt-in via config.use_face_id, defaulting to off.
        Any failure at any point here - disabled, package missing, no face
        detected, embedding format mismatch - falls back to the standard
        IP-Adapter composite path rather than dropping conditioning entirely.
        """
        if not self._ip_adapter_loaded:
            return {}

        if self._faceid_loaded and ip_adapter_refs and len(ip_adapter_refs) == 1:
            path = next(iter(ip_adapter_refs.values()))
            embeds = self._extract_face_embedding(path)
            if embeds is not None:
                self.pipeline.set_ip_adapter_scale(0.6)
                return {"ip_adapter_image_embeds": [embeds]}
            # No face found / extraction failed - fall through below.

        ip_image = self._build_ip_adapter_image(ip_adapter_refs)
        if ip_image is not None:
            self.pipeline.set_ip_adapter_scale(0.6)
            return {"ip_adapter_image": ip_image}
            
        import numpy as np
        dummy = Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8))
        self.pipeline.set_ip_adapter_scale(0.0)
        return {"ip_adapter_image": dummy}

    def _try_load_face_id(self):
        """
        Attempts to load insightface for face-embedding extraction, used
        alongside (not replacing) the regular IP-Adapter loaded in load().
        Gated behind config.use_face_id (default False). Any failure here
        - package not installed, model download failure, no compatible
        runtime - disables Face-ID for this run and generation continues
        on the standard IP-Adapter path exactly as before this feature existed.
        """
        try:
            from insightface.app import FaceAnalysis
            self._face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
            self._face_app.prepare(ctx_id=0, det_size=(640, 640))
            self._faceid_loaded = True
            logger.info("[Resource] Face-ID reference analysis ready (insightface loaded).")
        except Exception as e:
            self._faceid_loaded = False
            self._face_app = None
            logger.warning(
                f"[Resource] Face-ID unavailable ({e}) - character references will "
                "use standard IP-Adapter only. Install with: pip install insightface onnxruntime"
            )

    def _extract_face_embedding(self, image_path: str):
        """Returns a face embedding tensor for image_path, or None if unavailable/no face found."""
        if not self._faceid_loaded or self._face_app is None:
            return None
        try:
            import cv2
            import torch

            img = cv2.imread(image_path)
            if img is None:
                logger.warning(f"[Face-ID] Could not read {image_path}.")
                return None
            faces = self._face_app.get(img)
            if not faces:
                logger.info(f"[Face-ID] No face detected in {image_path} - using standard IP-Adapter instead.")
                return None
            return torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
        except Exception as e:
            logger.warning(f"[Face-ID] Embedding extraction failed for {image_path}: {e}")
            return None

    def capabilities(self):
        from plugins.interfaces import ProviderCapability
        return ProviderCapability(modality="image", supports_lora=bool(self.config.adapter))
        
    def load(self) -> None:
        if self.pipeline is not None:
            logger.info(f"[Resource] Model {self.config.model_id} already resident in VRAM.")
            return

        import torch
        from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel

        logger.info(f"[Resource] Loading {self.config.model_id} into VRAM...")
        
        unet = None
        if self.config.adapter and "SDXL-Lightning" in self.config.adapter:
            # Handle specific adapter logic
            unet = UNet2DConditionModel.from_pretrained(
                self.config.adapter,
                subfolder="unet",
                torch_dtype=getattr(torch, self.config.dtype, torch.float16),
                cache_dir=self.config.cache_dir
            )
        
        # Load the base model
        if unet:
            self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                self.config.model_id,
                unet=unet,
                torch_dtype=getattr(torch, self.config.dtype, torch.float16),
                variant="fp16",
                use_safetensors=True,
                cache_dir=self.config.cache_dir
            )
        else:
            self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                self.config.model_id,
                torch_dtype=getattr(torch, self.config.dtype, torch.float16),
                variant="fp16",
                use_safetensors=True,
                cache_dir=self.config.cache_dir
            )
            
        if self.device == "cuda":
            if self.config.cpu_offload:
                self.pipeline.enable_model_cpu_offload()
            else:
                self.pipeline.to("cuda")

        try:
            self.pipeline.load_ip_adapter(
                "h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin"
            )
            self.pipeline.set_ip_adapter_scale(0.6)
            self._ip_adapter_loaded = True
            logger.info("[Resource] IP-Adapter loaded - character reference conditioning active.")
        except Exception as e:
            self._ip_adapter_loaded = False
            logger.warning(
                f"[Resource] IP-Adapter failed to load ({e}) - continuing with "
                "text-only generation (no character reference conditioning)."
            )

        if getattr(self.config, "use_face_id", False):
            self._try_load_face_id()

        self.warmup()
        
    def warmup(self) -> None:
        logger.info("[Resource] Warming up model...")
        from core.domain.rendering.presets import RenderPreset
        preset = RenderPreset(width=256, height=256, steps=1, cfg=0.0)
        self._generate_internal("warmup", "", 42, preset)
        
    def _generate_internal(self, prompt, negative, seed, preset, callback=None):
        import torch
        from diffusers import EulerDiscreteScheduler

        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Instantiate scheduler from string
        if preset.sampler == "euler":
            self.pipeline.scheduler = EulerDiscreteScheduler.from_config(
                self.pipeline.scheduler.config, 
                timestep_spacing="trailing"
            )
        # We can add more samplers later
        
        def cb_wrapper(step, timestep, latents):
            if callback:
                callback(step, preset.steps)
                
        kwargs = dict(
            prompt=prompt,
            negative_prompt=negative,
            num_inference_steps=preset.steps,
            guidance_scale=preset.cfg,
            width=preset.width,
            height=preset.height,
            generator=generator,
            callback=cb_wrapper if callback else None,
            callback_steps=1
        )
        if self._ip_adapter_loaded:
            import numpy as np
            from PIL import Image
            dummy = Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8))
            self.pipeline.set_ip_adapter_scale(0.0)
            kwargs["ip_adapter_image"] = dummy

        image = self.pipeline(**kwargs).images[0]
        
        return image
        
    def health_check(self) -> ProviderHealth:
        import torch
        loaded = self.pipeline is not None
        vram = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
        return ProviderHealth(
            loaded=loaded, 
            device=self.device, 
            model=self.config.model_id, 
            dtype=self.config.dtype, 
            vram_allocated_gb=vram
        )
        
    def unload(self) -> None:
        logger.info(f"[Resource] Unloading {self.config.model_id}...")
        if self.pipeline:
            del self.pipeline
            self.pipeline = None
        from core.utils.vram import flush_vram
        flush_vram("Diffusion unloaded")

LocalDiffusionProvider = DiffusersProvider
