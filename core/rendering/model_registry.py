from typing import Dict, Type
from plugins.interfaces import ImageGenerationProvider
import logging

logger = logging.getLogger(__name__)

class ModelRegistry:
    """
    Registry mapping string identifiers (e.g. 'sdxl-lightning') to 
    provider factory classes to prevent hardcoding specific providers 
    throughout the compiler pipeline.
    """
    _providers: Dict[str, Type[ImageGenerationProvider]] = {}
    
    @classmethod
    def register(cls, model_id: str, provider_cls: Type[ImageGenerationProvider]):
        cls._providers[model_id] = provider_cls
        
    @classmethod
    def resolve(cls, model_id: str, **kwargs) -> ImageGenerationProvider:
        if model_id not in cls._providers:
            # Fallback to local diffusion provider if not found
            logger.warning(f"Model '{model_id}' not found in registry. Falling back to local diffusers.")
            from plugins.local_diffusion import DiffusersProvider
            return DiffusersProvider(**kwargs)
            
        provider_cls = cls._providers[model_id]
        return provider_cls(**kwargs)

# Pre-register known providers
try:
    from plugins.local_diffusion import DiffusersProvider, MockProvider
    ModelRegistry.register("sdxl-lightning", DiffusersProvider)
    ModelRegistry.register("mock", MockProvider)
except ImportError:
    pass
