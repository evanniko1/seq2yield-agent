"""Deterministic, keyless provider (adopted from shepherd's deterministic provider — see
docs/DECISIONS.md). Given a request, it returns a REPRODUCIBLE, schema-valid structured object
derived from a hash of the prompt — no network, no API key, no randomness across runs. This makes
council decisions REPLAYABLE (an infra-audit gap: real LLM decisions were not reproducible) and lets
the full council loop + the live-shaped tests run offline in CI.

It does not simulate a good agent — it fills the requested pydantic schema with valid, constrained,
hash-seeded values (respecting enums, numeric bounds, min-length lists, and a few field-name
heuristics so validators like `dataset must be registered` pass). Every call is still logged as a
$0 ModelCallRecord, so the trace/cost accounting is uniform with real providers.
"""
from __future__ import annotations

import hashlib
from typing import Type

import numpy as np
from pydantic import BaseModel

from .base import BaseStructuredClient

# field-name → a value known to satisfy the council schemas' validators
_HINTS = {
    "dataset": "ecoli", "comparator_model": "rf", "base_model": "cnn", "model_family": "cnn",
    "feature_set": "one_hot", "sampling_policy": "random", "feature_scaling": "none",
    "subregion": "all", "scope": "global", "primary_metric": "r2",
}


def _seed(system: str, user: str) -> int:
    return int(hashlib.sha256((system + "\x00" + user).encode()).hexdigest()[:8], 16)


def _fill(node: dict, defs: dict, rng: np.random.Generator, name: str | None = None):
    if "$ref" in node:
        return _fill(defs[node["$ref"].split("/")[-1]], defs, rng, name)
    if "anyOf" in node:
        opts = [o for o in node["anyOf"] if o.get("type") != "null"]
        return _fill(opts[0], defs, rng, name) if opts else None
    if "enum" in node:
        enum = node["enum"]
        if name in _HINTS and _HINTS[name] in enum:          # honour hints for enums too, so e.g.
            return _HINTS[name]                              # model_family=cnn / comparator=rf never
        if name == "intervention_type" and "model_architecture" in enum:   # collide (self-comparison)
            return "model_architecture"
        return enum[int(rng.integers(0, len(enum)))]
    t = node.get("type")
    if t == "object":
        req = set(node.get("required", []))
        props = node.get("properties", {})
        return {k: _fill(v, defs, rng, k) for k, v in props.items() if k in req}
    if t == "array":
        n = max(int(node.get("minItems", 0)), 0)
        item = node.get("items", {"type": "string"})
        return [_fill(item, defs, rng, name) for _ in range(n)]
    if t == "integer":
        lo, hi = int(node.get("minimum", 1)), int(node.get("maximum", 5))
        return int(rng.integers(lo, hi + 1)) if hi >= lo else lo
    if t == "number":
        lo, hi = float(node.get("minimum", 0.0)), float(node.get("maximum", 1.0))
        return round(float(rng.uniform(lo, min(hi, lo + 1))), 3)
    if t == "boolean":
        return False
    if t == "null":
        return None
    # string
    if name in _HINTS:
        return _HINTS[name]
    return f"det_{name or 'x'}"


class DeterministicClient(BaseStructuredClient):
    provider = "deterministic"

    def __init__(self, model: str = "det-v1"):
        self.model = model

    @staticmethod
    def available() -> bool:
        return True

    def _raw(self, *, system, user, schema: Type[BaseModel], temperature, max_tokens):
        js = schema.model_json_schema()
        defs = js.get("$defs", {})
        rng = np.random.default_rng(_seed(system, user))
        data = _fill(js, defs, rng)
        # a second pass instance to ensure it validates (raises here if the filler missed a field)
        schema.model_validate(data)
        return data, {"input": len(system) + len(user), "output": 0}
