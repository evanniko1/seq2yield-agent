"""Paired bootstrap over per-series R² deltas (docs/CONTRACTS.md §9, statistics autonomy).

Compares a candidate against a baseline using the paired per-series differences in R²
(candidate - baseline), resampling series with replacement to get a CI on the mean delta.
This is conservative and respects the paired structure (same series, same splits).
"""
from __future__ import annotations

import numpy as np


def paired_bootstrap_ci(baseline_per_series, candidate_per_series, *,
                        n_boot: int = 10000, alpha: float = 0.05, seed: int = 0) -> dict:
    base = np.asarray(baseline_per_series, dtype=float)
    cand = np.asarray(candidate_per_series, dtype=float)
    if base.shape != cand.shape:
        raise ValueError("baseline and candidate must be paired (same series order/length)")
    deltas = cand - base
    rng = np.random.default_rng(seed)
    n = len(deltas)
    boot_means = np.array([rng.choice(deltas, size=n, replace=True).mean()
                           for _ in range(n_boot)])
    lo, hi = np.quantile(boot_means, [alpha / 2, 1 - alpha / 2])
    return {
        "mean_delta": float(deltas.mean()),
        "ci": [float(lo), float(hi)],
        "alpha": alpha,
        "n_series": int(n),
        "excludes_zero": bool(lo > 0 or hi < 0),
    }
