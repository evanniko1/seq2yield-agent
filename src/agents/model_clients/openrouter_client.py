"""OpenRouter client — OpenAI-compatible Chat Completions with json_schema response format.

Diversity-class provider (hosted model catalogue). Requires OPENROUTER_API_KEY. Reuses the
OpenAI client since the API surface is compatible; only base URL + key env differ.
"""
from __future__ import annotations

from .openai_client import OpenAIClient

_OPENROUTER_BASE = "https://openrouter.ai/api/v1"


class OpenRouterClient(OpenAIClient):
    provider = "openrouter"

    def __init__(self, model: str, api_key_env: str = "OPENROUTER_API_KEY",
                 base_url: str = _OPENROUTER_BASE, max_retries: int = 2, timeout: float = 120.0):
        super().__init__(model, api_key_env=api_key_env, base_url=base_url,
                         max_retries=max_retries, timeout=timeout,
                         extra_headers={"HTTP-Referer": "https://github.com/evanniko1/seq2yield-agent",
                                        "X-Title": "seq2yield-agent"})
