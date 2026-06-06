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


def _r2(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot else float("nan")


def bootstrap_r2_ci(y_true, y_pred, *, n_boot: int = 2000, alpha: float = 0.05,
                    seed: int = 0) -> dict:
    """CI on a single model's R² by resampling TEST SEQUENCES (for pooled datasets like yeast,
    where per-group counts are too small for per-group R²)."""
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    n = len(y_true)
    rng = np.random.default_rng(seed)
    boot = np.array([_r2(y_true[idx], y_pred[idx])
                     for idx in (rng.integers(0, n, n) for _ in range(n_boot))])
    lo, hi = np.nanquantile(boot, [alpha / 2, 1 - alpha / 2])
    return {"r2": _r2(y_true, y_pred), "ci": [float(lo), float(hi)], "n": int(n)}


def paired_bootstrap_r2(y_true, pred_candidate, pred_baseline, *, n_boot: int = 2000,
                        alpha: float = 0.05, seed: int = 0) -> dict:
    """Paired CI on ΔR² (candidate - baseline) by resampling the SAME test sequences for both
    models (sequence-level paired bootstrap; the pooled-dataset analog of the per-series test)."""
    y_true = np.asarray(y_true, float)
    pc = np.asarray(pred_candidate, float)
    pb = np.asarray(pred_baseline, float)
    n = len(y_true)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        deltas.append(_r2(y_true[idx], pc[idx]) - _r2(y_true[idx], pb[idx]))
    deltas = np.array(deltas)
    lo, hi = np.nanquantile(deltas, [alpha / 2, 1 - alpha / 2])
    return {"mean_delta": float(_r2(y_true, pc) - _r2(y_true, pb)),
            "ci": [float(lo), float(hi)], "n": int(n),
            "excludes_zero": bool(lo > 0 or hi < 0)}
