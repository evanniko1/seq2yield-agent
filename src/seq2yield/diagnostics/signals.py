"""Deterministic diagnostic signals (K4). Pure functions over arrays / per-size curves — no
model state, no randomness — so they are reproducible and trusted (the harness computes them).
"""
from __future__ import annotations

import numpy as np

from ..training import metrics as M


def _arr(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def generalization_gap(y_train, pred_train, y_test, pred_test) -> dict:
    """Overfit signal: train R² minus test R². A large positive gap = the model fits training
    structure that does not generalize."""
    tr, te = M.r2(_arr(y_train), _arr(pred_train)), M.r2(_arr(y_test), _arr(pred_test))
    return {"train_r2": round(float(tr), 4), "test_r2": round(float(te), 4),
            "gap": round(float(tr - te), 4)}


def calibration(y_true, y_pred) -> dict:
    """Regression calibration: OLS slope/intercept of actual ~ predicted. Slope≈1, intercept≈0
    is well-calibrated; slope<1 means the model's spread is over-confident (regression to mean)."""
    yt, yp = _arr(y_true), _arr(y_pred)
    if len(yt) < 3 or np.std(yp) < 1e-9:
        return {"slope": None, "intercept": None}
    slope, intercept = np.polyfit(yp, yt, 1)
    return {"slope": round(float(slope), 4), "intercept": round(float(intercept), 4)}


def residual_diagnostics(y_true, y_pred) -> dict:
    """Residual structure: mean residual (prediction bias) and heteroscedasticity (correlation of
    |residual| with the prediction — error growing with magnitude breaks homoscedasticity)."""
    yt, yp = _arr(y_true), _arr(y_pred)
    resid = yt - yp
    hetero = None
    if len(resid) > 2 and np.std(yp) > 1e-9 and np.std(np.abs(resid)) > 1e-9:
        hetero = float(np.corrcoef(np.abs(resid), yp)[0, 1])
    return {"mean_residual": round(float(resid.mean()), 4),
            "residual_std": round(float(resid.std()), 4),
            "heteroscedasticity_corr": (round(hetero, 4) if hetero is not None else None)}


def _ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample Kolmogorov–Smirnov statistic (max CDF gap); no scipy dependency."""
    a, b = np.sort(a), np.sort(b)
    grid = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, grid, side="right") / len(a)
    cdf_b = np.searchsorted(b, grid, side="right") / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def split_representativeness(y_train, y_test) -> dict:
    """Is the held-out test target distribution representative of train? Standardized mean shift,
    std ratio, and KS statistic. A tail-sliced / non-stratified split shows up as a large shift
    (the exact failure mode of a naive 'last-X%' validation split)."""
    yt, ye = _arr(y_train), _arr(y_test)
    sd = yt.std() or 1.0
    return {"mean_shift_std": round(float((ye.mean() - yt.mean()) / sd), 4),
            "std_ratio": round(float((ye.std() or 0.0) / sd), 4),
            "ks": round(_ks_statistic(yt, ye), 4)}


def sequence_leakage(train_seqs, test_seqs) -> dict:
    """Exact-duplicate leakage: fraction of test sequences that also appear in the training set."""
    tr = set(map(str, train_seqs))
    test = list(map(str, test_seqs))
    leaked = sum(1 for s in test if s in tr)
    return {"n_test": len(test), "n_leaked": int(leaked),
            "leak_frac": round(leaked / max(1, len(test)), 4)}


def target_coverage(y_train, y_test) -> dict:
    """Extrapolation: fraction of test targets outside the train target range (the model is asked
    to predict beyond what it saw)."""
    yt, ye = _arr(y_train), _arr(y_test)
    lo, hi = yt.min(), yt.max()
    extrap = float(np.mean((ye < lo) | (ye > hi))) if len(ye) else 0.0
    return {"train_range": [round(float(lo), 3), round(float(hi), 3)],
            "extrapolated_frac": round(extrap, 4)}


def learning_curve_shape(per_size) -> dict:
    """From the per-size verdicts: is the candidate still improving at the largest train size
    (=> data-limited, more data would help) or has it plateaued?"""
    if not per_size:
        return {"still_improving": None, "candidate_by_size": {}}
    ordered = sorted(per_size, key=lambda p: p["train_size"])
    by_size = {p["train_size"]: round(float(p.get("candidate_mean", 0.0)), 4) for p in ordered}
    still = None
    if len(ordered) >= 2:
        last_gain = ordered[-1].get("candidate_mean", 0.0) - ordered[-2].get("candidate_mean", 0.0)
        still = bool(last_gain > 0.01)
    return {"still_improving": still, "candidate_by_size": by_size,
            "last_gain": (round(float(last_gain), 4) if len(ordered) >= 2 else None)}
