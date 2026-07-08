"""
Cloud LLM Provider
====================
Thin wrapper around a cloud LLM's chat completion endpoint, used ONLY as an
optional fallback in the LLM chain if the local model fails to load.

Never used by default, never required, and never contacted unless:
  1. The local provider already failed to load, AND
  2. A corresponding API key is present in the environment.

This is deliberately minimal - no SDK dependency, just `requests` against
each provider's documented REST endpoint - to keep the "optional fallback"
surface small and easy to audit.

NOTE ON MODEL NAMES: the default model per provider is set to a small, fast,
cost-efficient option suitable for structured JSON extraction (not creative
writing). Both are overridable via environment variables, since model
identifiers change over time and the ones baked in here may go stale -
check the provider's current model list if generation fails with a
"model not found"-style error.
"""

import json
import logging
import os

from plugins.json_utils import extract_json

logger = logging.getLogger(__name__)


class CloudLLMProvider:
    CONFIGS = {
        "anthropic": {
            "env_key": "ANTHROPIC_API_KEY",
            "env_model": "NOVELFACTORY_ANTHROPIC_MODEL",
            "default_model": "claude-haiku-4-5-20251001",
            "url": "https://api.anthropic.com/v1/messages",
        },
        "openai": {
            "env_key": "OPENAI_API_KEY",
            "env_model": "NOVELFACTORY_OPENAI_MODEL",
            "default_model": "gpt-4o-mini",
            "url": "https://api.openai.com/v1/chat/completions",
        },
    }

    def __init__(self, provider: str = "anthropic"):
        if provider not in self.CONFIGS:
            raise ValueError(f"Unknown cloud provider '{provider}'. Choices: {list(self.CONFIGS)}")
        self.provider = provider
        self.cfg = self.CONFIGS[provider]
        self.api_key = None
        self.model = self.cfg["default_model"]

    def load(self):
        """
        Only 'loading' step is confirming an API key is present - there's no
        local resource to allocate. Raises if the key is missing, so the
        fallback chain correctly skips to the next provider rather than
        attempting (and failing) a network call with no credentials.
        """
        self.api_key = os.environ.get(self.cfg["env_key"])
        if not self.api_key:
            raise RuntimeError(
                f"{self.cfg['env_key']} is not set - cannot use {self.provider} as a fallback."
            )
        self.model = os.environ.get(self.cfg["env_model"], self.cfg["default_model"])
        logger.info(f"[CloudLLM] Ready to use {self.provider} ({self.model}) as a fallback if needed.")

    def generate_json(self, prompt: str, schema: dict) -> dict:
        import requests

        url, headers, body = self._build_request(prompt)
        resp = requests.post(url, headers=headers, json=body, timeout=120)
        resp.raise_for_status()

        text = self._parse_response(resp.json())
        json_text = extract_json(text)
        result = json.loads(json_text)

        missing_keys = [k for k in schema.keys() if k not in result]
        if missing_keys:
            raise ValueError(f"Extracted JSON is missing required schema keys: {missing_keys}")
        return result

    def unload(self):
        pass  # no local resources to free

    # ── Request/response building - separated out so they're testable ────────
    # without needing to make a real network call. ─────────────────────────────

    def _build_request(self, prompt: str):
        if self.provider == "anthropic":
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
            body = {
                "model": self.model,
                "max_tokens": 4096,
                "messages": [{"role": "user", "content": prompt}],
            }
            return self.cfg["url"], headers, body

        if self.provider == "openai":
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "content-type": "application/json",
            }
            body = {
                "model": self.model,
                "max_tokens": 4096,
                "messages": [{"role": "user", "content": prompt}],
            }
            return self.cfg["url"], headers, body

        raise ValueError(f"Unhandled provider '{self.provider}'.")

    def _parse_response(self, response_json: dict) -> str:
        if self.provider == "anthropic":
            return response_json["content"][0]["text"]
        if self.provider == "openai":
            return response_json["choices"][0]["message"]["content"]
        raise ValueError(f"Unhandled provider '{self.provider}'.")
