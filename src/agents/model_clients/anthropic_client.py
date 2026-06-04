"""Anthropic client — structured output via forced tool use.

Authority-class provider. Anthropic's most reliable structured-output mechanism is a forced
tool call whose input_schema is the target pydantic schema; the tool input is the result.
Requires ANTHROPIC_API_KEY.
"""
from __future__ import annotations

import os
from typing import Type

from pydantic import BaseModel

from .base import BaseStructuredClient, ProviderUnavailable

_TOOL = "emit_result"


class AnthropicClient(BaseStructuredClient):
    provider = "anthropic"

    def __init__(self, model: str, api_key_env: str = "ANTHROPIC_API_KEY", max_retries: int = 2):
        self.model = model
        self.api_key = os.environ.get(api_key_env)
        self.max_retries = max_retries
        self._client = None

    def _ensure(self):
        if not self.api_key:
            raise ProviderUnavailable(f"{self.provider}: API key not set")
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def _raw(self, *, system, user, schema: Type[BaseModel], temperature, max_tokens):
        client = self._ensure()
        tool = {"name": _TOOL,
                "description": f"Return a {schema.__name__} object.",
                "input_schema": schema.model_json_schema()}
        resp = client.messages.create(
            model=self.model, max_tokens=max_tokens, temperature=temperature,
            system=system, messages=[{"role": "user", "content": user}],
            tools=[tool], tool_choice={"type": "tool", "name": _TOOL})
        usage = {"input": resp.usage.input_tokens, "output": resp.usage.output_tokens}
        for block in resp.content:
            if getattr(block, "type", None) == "tool_use" and block.name == _TOOL:
                return block.input, usage          # dict -> validated by base
        raise RuntimeError("anthropic: no tool_use block returned")

    def available(self) -> bool:
        return bool(self.api_key)
