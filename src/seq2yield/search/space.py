"""C2 — sampling and perturbation over the C1 SEARCH_SPACE.

`sample_config` draws a random point (random-search exploration); `perturb_config` nudges an
incumbent (the local-search acquisition used in the exploitation phase — the "results fed back as
the next acquisition step" of the hybrid engine). All draws go back through the registry's
`clean_hyperparameters`, so a sampled/perturbed config is always a valid, coercible point of the
model's whitelist — no invalid config ever reaches training.
"""
from __future__ import annotations

import math

import numpy as np

from ..models import registry as reg


def _sample_scalar(meta: dict, rng: np.random.Generator):
    t = meta["type"]
    if t == "int":
        lo, hi = meta["range"]
        return int(rng.integers(int(lo), int(hi) + 1))
    if t == "float":
        lo, hi = meta["range"]
        return float(rng.uniform(lo, hi))
    if t == "log_float":
        lo, hi = meta["range"]
        return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))
    if t == "bool":
        return bool(rng.integers(0, 2))
    if t == "categorical":
        return meta["choices"][int(rng.integers(0, len(meta["choices"])))]
    raise ValueError(f"unhandled scalar type {t!r}")


def _sample_list(meta: dict, rng: np.random.Generator):
    lo, hi = meta["range"]
    llo, lhi = meta["len"]
    n = int(rng.integers(int(llo), int(lhi) + 1)) if lhi > llo else int(llo)
    n = max(n, 1)
    if meta["type"] == "int_list":
        return [int(rng.integers(int(lo), int(hi) + 1)) for _ in range(n)]
    return [float(rng.uniform(lo, hi)) for _ in range(n)]


def sample_value(meta: dict, rng: np.random.Generator):
    return _sample_list(meta, rng) if meta["type"].endswith("list") else _sample_scalar(meta, rng)


def sample_config(model: str, rng: np.random.Generator, knobs=None) -> dict:
    """A random, valid config over `knobs` (default: the model's whole space)."""
    space = reg.search_space(model)
    keys = list(space) if knobs is None else [k for k in knobs if k in space]
    raw = {k: sample_value(space[k], rng) for k in keys}
    return reg.clean_hyperparameters(model, raw)


def _perturb_value(v, meta: dict, rng: np.random.Generator):
    t = meta["type"]
    if t in ("int", "float", "log_float"):
        lo, hi = meta["range"]
        span = hi - lo
        step = rng.normal(0, 0.15 * span)                  # ~15% of the range
        nv = float(v) + step
        nv = min(max(nv, lo), hi)
        return int(round(nv)) if t == "int" else float(nv)
    if t.endswith("list"):
        lst = list(v)
        if lst:
            i = int(rng.integers(0, len(lst)))
            lst[i] = _perturb_value(lst[i], {"type": t[:-5], "range": meta["range"]}, rng)
        return lst
    return _sample_scalar(meta, rng)                        # bool/categorical: resample


def perturb_config(model: str, config: dict, rng: np.random.Generator, prob: float = 0.34) -> dict:
    """Nudge an incumbent: each present knob is perturbed independently w.p. `prob` (>=1 always).
    The neighbourhood is the local-search acquisition around the current best."""
    space = reg.search_space(model)
    keys = [k for k in config if k in space]
    if not keys:
        return dict(config)
    picks = [k for k in keys if rng.random() < prob]
    if not picks:
        picks = [keys[int(rng.integers(0, len(keys)))]]     # perturb at least one knob
    out = dict(config)
    for k in picks:
        out[k] = _perturb_value(config[k], space[k], rng)
    return reg.clean_hyperparameters(model, out)
