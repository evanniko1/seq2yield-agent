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

    def _raw(self, *, system, user, schema: Type[BaseModel], temperature, max_tokens,
             cache: bool = True):
        client = self._ensure()
        tool = {"name": _TOOL,
                "description": f"Return a {schema.__name__} object.",
                "input_schema": schema.model_json_schema()}
        # Prompt caching (Anthropic ephemeral breakpoints). Two large static prefixes recur across
        # the council's ~11 calls per cycle: the tool schema is identical for a given output type
        # (every reviewer emits CouncilReviewItem), and the system prompt is identical across all
        # calls for a given role. Marking both caches the prefix, so repeated same-schema /
        # same-role calls (reviewers over N proposals, debate rounds, later cycles within the TTL)
        # read it at ~10% of the input cost. Cache read/write tokens are surfaced in usage so the
        # telemetry shows caching working. Below the model's min cacheable length Anthropic simply
        # ignores the marker — safe to always set.
        system_param = system
        if cache:
            tool["cache_control"] = {"type": "ephemeral"}
            if system:
                system_param = [{"type": "text", "text": system,
                                 "cache_control": {"type": "ephemeral"}}]
        resp = client.messages.create(
            model=self.model, max_tokens=max_tokens, temperature=temperature,
            system=system_param, messages=[{"role": "user", "content": user}],
            tools=[tool], tool_choice={"type": "tool", "name": _TOOL})
        u = resp.usage
        usage = {"input": u.input_tokens, "output": u.output_tokens,
                 "cache_read": getattr(u, "cache_read_input_tokens", 0) or 0,
                 "cache_write": getattr(u, "cache_creation_input_tokens", 0) or 0}
        for block in resp.content:
            if getattr(block, "type", None) == "tool_use" and block.name == _TOOL:
                return block.input, usage          # dict -> validated by base
        raise RuntimeError("anthropic: no tool_use block returned")

    def available(self) -> bool:
        return bool(self.api_key)
