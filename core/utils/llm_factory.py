"""
Shared LLM instantiation helper.

Replaces the four identical lazy-instantiation blocks that existed verbatim in:
  • core/planning/shot_planner.py
  • core/planning/cast_planner.py
  • core/planning/scene_splitter.py
  • core/planning/director_planner.py

Usage
-----
from core.utils.llm_factory import ensure_llm

class MyStage:
    def __init__(self, llm_provider=None):
        self.llm = llm_provider

    def execute(self, context):
        self.llm = ensure_llm(self.llm)   # lazy-load with defaults
        ...
"""
from __future__ import annotations
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from plugins.interfaces import LLMProvider


def ensure_llm(
    llm: Optional["LLMProvider"],
    model_id: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> "LLMProvider":
    """
    Return `llm` if already set; otherwise instantiate a `LocalLLMProvider`
    with the given `model_id` and `cache_dir`.

    Parameters
    ----------
    llm:
        An already-loaded provider, or None.
    model_id:
        HuggingFace model ID to pass to LocalLLMProvider.
        Defaults to the provider's own default when None.
    cache_dir:
        Directory for local model weights.
        Defaults to the provider's own default when None.

    Returns
    -------
    LLMProvider
        The existing provider unchanged, or a freshly created LocalLLMProvider.
    """
    if llm is not None:
        return llm

    import logging
    logger = logging.getLogger(__name__)

    kwargs = {}
    if model_id:
        kwargs["model_id"] = model_id
    if cache_dir:
        kwargs["cache_dir"] = cache_dir

    try:
        from plugins.local_llm import LocalLLMProvider
        provider = LocalLLMProvider(**kwargs)
        logger.info(f"[llm_factory] Instantiated LocalLLMProvider (model_id={model_id or 'default'})")
        return provider
    except Exception as e:
        logger.error(f"[llm_factory] Failed to instantiate LocalLLMProvider: {e}")
        raise
