"""
LLM Fallback Chain
===================
Tries a sequence of LLM providers in order, using the first one that loads
successfully for the rest of the pipeline run.

Local-first by design: NovelFactory's whole premise is free, local,
Kaggle-native generation. A cloud provider is only ever consulted if the
local model fails to load in the first place (network hiccup, HF rate limit,
VRAM exhaustion, etc.) - not as a default preference, and only if the
corresponding API key happens to be present in the environment. If nothing
in the chain loads, the failure is surfaced clearly rather than silently
producing empty results.
"""

import logging
from typing import Callable, List

logger = logging.getLogger(__name__)


class FallbackLLMProvider:
    def __init__(self, factories: List[Callable[[], object]]):
        """
        factories: ordered list of zero-arg callables, each returning an
        unloaded provider instance (something with .load()/.generate_json()/
        .unload()). Tried in order; the first one whose .load() succeeds is
        used for the remainder of this provider's lifetime.
        """
        if not factories:
            raise ValueError("FallbackLLMProvider needs at least one provider factory.")
        self.factories = factories
        self.active = None
        self.active_name = None

    def load(self):
        errors = []
        for i, factory in enumerate(self.factories):
            try:
                provider = factory()
            except Exception as e:
                errors.append(f"factory[{i}] construction failed: {e}")
                continue
            try:
                provider.load()
            except Exception as e:
                errors.append(f"{type(provider).__name__} failed to load: {e}")
                logger.warning(
                    f"[LLM Fallback] {type(provider).__name__} failed to load "
                    f"({e}) - trying next provider in the chain..."
                )
                continue

            self.active = provider
            self.active_name = type(provider).__name__
            if i > 0:
                logger.warning(
                    f"[LLM Fallback] Using {self.active_name} (fallback #{i}) "
                    "- the primary provider was unavailable."
                )
            else:
                logger.info(f"[LLM Fallback] Using {self.active_name} (primary).")
            return

        raise RuntimeError(
            "All LLM providers in the fallback chain failed to load:\n"
            + "\n".join(f"  - {e}" for e in errors)
        )

    def generate_json(self, prompt: str, schema: dict) -> dict:
        if self.active is None:
            raise RuntimeError("FallbackLLMProvider.load() must succeed before generate_json().")
        return self.active.generate_json(prompt, schema)

    def unload(self):
        if self.active is not None:
            self.active.unload()
