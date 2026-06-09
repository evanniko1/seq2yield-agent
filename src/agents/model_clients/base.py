"""Provider-agnostic model client foundation (docs/CONTRACTS.md §6, §7; AGENTS.md §5).

Every concrete client returns a validated pydantic object and logs a ModelCallRecord. The
base provides the structured-output finalize/retry loop and JSONL call logging so each
provider only implements the raw request.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, Type, runtime_checkable

from pydantic import BaseModel, ValidationError

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LOG = ROOT / "reports" / "model_calls.jsonl"


def _load_dotenv(path: Path = ROOT / ".env") -> None:
    """Load KEY=VALUE lines from a gitignored .env into the environment (real env vars win).
    Keeps API keys out of committed files (copy .env.example -> .env and fill in)."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if val and not os.environ.get(key):    # fill if unset OR present-but-empty (real non-empty wins)
            os.environ[key] = val


_load_dotenv()


class ModelCallRecord(BaseModel):
    provider: str
    model: str
    role: str
    prompt_hash: str
    prompt_template: str | None = None   # C11: which versioned template produced this call
    prompt_version: str | None = None    # C11: template version (drift vs intentional revision)
    run_id: str | None = None            # RL-trace: council trajectory id (join key)
    task_id: str | None = None           # RL-trace: question/cell being worked
    schema_name: str
    raw_text: str | None = None
    output_hash: str | None = None       # RL-trace: content ref to the output
    parsed: dict | None = None
    token_usage: dict | None = None
    cost_usd: float | None = None        # RL-trace: per-call $ (priced at log time)
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
        meta = metadata or {}
        tmpl = meta.get("prompt_template")
        tver = meta.get("prompt_version")
        try:                                             # RL-trace: tag with active trajectory
            from .. import trace
            _ctx = trace.current()
        except Exception:
            _ctx = {}
        rid, tid = _ctx.get("trajectory_id"), _ctx.get("task_id")
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
                    prompt_template=tmpl, prompt_version=tver, run_id=rid, task_id=tid,
                    schema_name=schema.__name__, raw_text=raw_text,
                    output_hash=_output_hash(raw_text), parsed=obj.model_dump(),
                    token_usage=usage, cost_usd=_call_cost(self.provider, self.model, usage),
                    latency_sec=round(time.perf_counter() - t0, 3),
                    retries=attempt, success=True,
                    ts=datetime.now(timezone.utc).isoformat())
                log_call(rec, log_path)
                return obj
            except ProviderUnavailable as e:
                rec = ModelCallRecord(
                    provider=self.provider, model=self.model, role=role, prompt_hash=ph,
                    prompt_template=tmpl, prompt_version=tver, run_id=rid, task_id=tid,
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
            prompt_template=tmpl, prompt_version=tver, run_id=rid, task_id=tid,
            schema_name=schema.__name__, raw_text=raw_text,
            output_hash=_output_hash(raw_text), parsed=None,
            latency_sec=round(time.perf_counter() - t0, 3),
            retries=self.max_retries, success=False, error=str(last_err)[:500],
            ts=datetime.now(timezone.utc).isoformat())
        log_call(rec, log_path)
        raise RuntimeError(f"{self.provider}:{self.model} failed structured call: {last_err}")


def _output_hash(text: str | None) -> str | None:
    if text is None:
        return None
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


def _call_cost(provider: str, model: str, usage: dict | None) -> float | None:
    """Price a call at log time (RL-trace). Lazy import keeps base.py dependency-light."""
    try:
        from orchestration import budget
        return round(budget.call_cost({"provider": provider, "model": model,
                                       "token_usage": usage or {}}), 6)
    except Exception:
        return None
