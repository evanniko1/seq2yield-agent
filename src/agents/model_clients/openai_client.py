"""OpenAI client — structured output via the Chat Completions json_schema response format.

Implemented over direct HTTP (httpx) to avoid coupling to a specific openai SDK version.
Authority-class provider. Requires OPENAI_API_KEY.
"""
from __future__ import annotations

import os
from typing import Type

import httpx
from pydantic import BaseModel

from .base import BaseStructuredClient, ProviderUnavailable

_OPENAI_BASE = "https://api.openai.com/v1"


class OpenAIClient(BaseStructuredClient):
    provider = "openai"

    def __init__(self, model: str, api_key_env: str = "OPENAI_API_KEY",
                 base_url: str = _OPENAI_BASE, max_retries: int = 2, timeout: float = 120.0,
                 extra_headers: dict | None = None):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = os.environ.get(api_key_env)
        self.max_retries = max_retries
        self.timeout = timeout
        self.extra_headers = extra_headers or {}

    def _raw(self, *, system, user, schema: Type[BaseModel], temperature, max_tokens):
        if not self.api_key:
            raise ProviderUnavailable(f"{self.provider}: API key not set")
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": schema.__name__, "strict": True,
                                "schema": schema.model_json_schema()},
            },
        }
        headers = {"Authorization": f"Bearer {self.api_key}", **self.extra_headers}
        r = httpx.post(f"{self.base_url}/chat/completions", json=payload, headers=headers,
                       timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        text = data["choices"][0]["message"]["content"]
        return text, data.get("usage")

    def available(self) -> bool:
        return bool(self.api_key)
