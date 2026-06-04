"""Provider-agnostic model client foundation (docs/CONTRACTS.md §6, §7; AGENTS.md §5).

Every concrete client returns a validated pydantic object and logs a ModelCallRecord. The
base provides the structured-output finalize/retry loop and JSONL call logging so each
provider only implements the raw request.
"""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, Type, runtime_checkable

from pydantic import BaseModel, ValidationError

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LOG = ROOT / "reports" / "model_calls.jsonl"


class ModelCallRecord(BaseModel):
    provider: str
    model: str
    role: str
    prompt_hash: str
    schema_name: str
    raw_text: str | None = None
    parsed: dict | None = None
    token_usage: dict | None = None
    latency_sec: float = 0.0
    retries: int = 0
    success: bool = False
    error: str | None = None
    ts: str = ""


class ProviderUnavailable(RuntimeError):
    """Raised when a provider cannot be used (missing key, service down)."""


@runtime_checkable
class ModelClient(Protocol):
    provider: str
    model: str

    def complete_structured(self, *, system: str, user: str, schema: Type[BaseModel],
                            temperature: float = 0.2, max_tokens: int = 4096,
                            role: str = "unknown", metadata: dict | None = None) -> BaseModel:
        ...


def prompt_hash(system: str, user: str) -> str:
    return hashlib.sha256((system + "\x00" + user).encode("utf-8")).hexdigest()


def log_call(record: ModelCallRecord, path: str | Path = DEFAULT_LOG) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(record.model_dump_json() + "\n")


class BaseStructuredClient:
    """Mixin implementing the validate + retry + log loop around a raw request.

    Subclasses implement `_raw(system, user, schema, temperature, max_tokens) -> (text, usage)`
    returning a JSON string (or dict) the schema is validated against.
    """

    provider: str = "base"
    model: str = "unset"
    max_retries: int = 2

    def _raw(self, *, system: str, user: str, schema: Type[BaseModel],
             temperature: float, max_tokens: int):  # -> tuple[str|dict, dict|None]
        raise NotImplementedError

    def complete_structured(self, *, system: str, user: str, schema: Type[BaseModel],
                            temperature: float = 0.2, max_tokens: int = 4096,
                            role: str = "unknown", metadata: dict | None = None,
                            log_path: str | Path = DEFAULT_LOG) -> BaseModel:
        ph = prompt_hash(system, user)
        last_err = None
        raw_text = None
        t0 = time.perf_counter()
        for attempt in range(self.max_retries + 1):
            try:
                raw, usage = self._raw(system=system, user=user, schema=schema,
                                       temperature=temperature, max_tokens=max_tokens)
                raw_text = raw if isinstance(raw, str) else json.dumps(raw)
                data = json.loads(raw_text) if isinstance(raw, str) else raw
                obj = schema.model_validate(data)
                rec = ModelCallRecord(
                    provider=self.provider, model=self.model, role=role, prompt_hash=ph,
                    schema_name=schema.__name__, raw_text=raw_text, parsed=obj.model_dump(),
                    token_usage=usage, latency_sec=round(time.perf_counter() - t0, 3),
                    retries=attempt, success=True,
                    ts=datetime.now(timezone.utc).isoformat())
                log_call(rec, log_path)
                return obj
            except ProviderUnavailable as e:
                rec = ModelCallRecord(
                    provider=self.provider, model=self.model, role=role, prompt_hash=ph,
                    schema_name=schema.__name__, success=False, error=str(e)[:500],
                    latency_sec=round(time.perf_counter() - t0, 3),
                    ts=datetime.now(timezone.utc).isoformat())
                log_call(rec, log_path)
                raise                                # surface availability cleanly
            except (ValidationError, json.JSONDecodeError) as e:
                last_err = e
                # reinforce schema on retry
                user = (user + "\n\nReturn ONLY valid JSON matching the schema. "
                        f"Previous error: {str(e)[:300]}")
        rec = ModelCallRecord(
            provider=self.provider, model=self.model, role=role, prompt_hash=ph,
            schema_name=schema.__name__, raw_text=raw_text, parsed=None,
            latency_sec=round(time.perf_counter() - t0, 3),
            retries=self.max_retries, success=False, error=str(last_err)[:500],
            ts=datetime.now(timezone.utc).isoformat())
        log_call(rec, log_path)
        raise RuntimeError(f"{self.provider}:{self.model} failed structured call: {last_err}")
