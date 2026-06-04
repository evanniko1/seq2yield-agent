"""Ollama client — local structured output via the native /api/chat `format` schema.

Diversity-class provider (local, no API key). Works against any installed Ollama model.
"""
from __future__ import annotations

from typing import Type

import httpx
from pydantic import BaseModel

from .base import BaseStructuredClient, ProviderUnavailable


class OllamaClient(BaseStructuredClient):
    provider = "ollama"

    def __init__(self, model: str, base_url: str = "http://localhost:11434",
                 max_retries: int = 2, timeout: float = 120.0):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self.timeout = timeout

    def _raw(self, *, system, user, schema: Type[BaseModel], temperature, max_tokens):
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "format": schema.model_json_schema(),   # JSON-schema structured output
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        try:
            r = httpx.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout)
            r.raise_for_status()
        except (httpx.ConnectError, httpx.ConnectTimeout) as e:
            raise ProviderUnavailable(f"ollama unreachable at {self.base_url}: {e}") from e
        data = r.json()
        text = data["message"]["content"]
        usage = {"input": data.get("prompt_eval_count"), "output": data.get("eval_count")}
        return text, usage

    @staticmethod
    def available(base_url: str = "http://localhost:11434") -> bool:
        try:
            httpx.get(f"{base_url.rstrip('/')}/api/tags", timeout=3).raise_for_status()
            return True
        except Exception:
            return False
