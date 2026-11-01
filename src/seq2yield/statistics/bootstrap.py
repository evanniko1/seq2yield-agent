"""Paired bootstrap over per-series R² deltas (docs/CONTRACTS.md §9, statistics autonomy).

Compares a candidate against a baseline using the paired per-series differences in R²
(candidate - baseline), resampling series with replacement to get a CI on the mean delta.
This is conservative and respects the paired structure (same series, same splits).

THE RESAMPLING UNIT IS THE WHOLE ARGUMENT. A bootstrap CI is only as honest as its unit: the
unit must be something that was drawn independently. `paired_bootstrap_ci` resamples per-series
R² values, so its unit is a mutational series — already a cluster. The pooled functions resample
test rows, whose unit is one sequence, and that assumption fails on a DESIGNED library: SNV
pairs, motif-perturbation sets, tiling ladders and mutational series put near-copies in the same
test set, so a 4-member family contributes 4 correlated rows and the CI comes out too narrow
(audit ST-7, the follow-up to the grouped-split fix SA-1/ML-03). Pass `groups=` — the design
family of each test row — and they resample whole FAMILIES instead (a cluster bootstrap), which
is the unit that was actually drawn. Without `groups` the behaviour is unchanged, bit for bit,
so older runs still reproduce; the returned `unit` / `n_units` say which one was used, and the
caller records it as the C3 bootstrap_unit so a family-unit CI is never mixed with a
sequence-unit one.
"""
from __future__ import annotations

import numpy as np


def _units(groups, n: int) -> list[np.ndarray]:
    """Row indices per resampling unit, in order of first appearance.

    The order is load-bearing: when every label is distinct the units ARE the rows in their
    original order, so drawing units reduces exactly to drawing rows and the grouped path is a
    strict generalization of the sequence-level one (pinned by a test)."""
    labels = np.asarray(groups)
    if labels.ndim != 1 or labels.shape[0] != n:
        raise ValueError(f"groups must be one label per observation ({labels.shape} for n={n})")
    order: list = []
    members: dict = {}
    for i, lab in enumerate(labels.tolist()):
        if lab is None or (isinstance(lab, float) and np.isnan(lab)):
            raise ValueError("groups contains a missing label; an unlabelled row is neither its "
                             "own unit nor a sibling of the rest, so the caller must decide")
        if lab not in members:
            members[lab] = []
            order.append(lab)
        members[lab].append(i)
    return [np.asarray(members[k], dtype=int) for k in order]


def _draw(rng, n: int, units: list[np.ndarray] | None) -> np.ndarray:
    """One bootstrap resample: row indices (unit=sequence) or whole units (cluster bootstrap)."""
    if units is None:
        return rng.integers(0, n, n)
    return np.concatenate([units[k] for k in rng.integers(0, len(units), len(units))])


def _no_inference(n_units: int) -> dict:
    """Too few units to bootstrap. Fail loud and safe: no CI, nothing significant — a 1-unit
    cluster bootstrap would otherwise redraw the identical sample every time, report zero
    variance, and declare any non-zero delta significant."""
    return {"ci": [float("nan"), float("nan")], "excludes_zero": False, "p_value": 1.0,
            "note": f"{n_units} resampling unit(s): between-unit variance is not estimable, "
                    "so no CI is reported and the comparison stays inconclusive"}


def paired_bootstrap_ci(baseline_per_series, candidate_per_series, *,
                        n_boot: int = 10000, alpha: float = 0.05, seed: int = 0) -> dict:
    base = np.asarray(baseline_per_series, dtype=float)
    cand = np.asarray(candidate_per_series, dtype=float)
    if base.shape != cand.shape:
        raise ValueError("baseline and candidate must be paired (same series order/length)")
    deltas = cand - base
    n = len(deltas)
    out = {"mean_delta": float(deltas.mean()), "alpha": alpha, "n_series": int(n),
           "unit": "series", "n_units": int(n)}
    if n < 2:                       # the unit here is already a cluster; one of them proves nothing
        return out | _no_inference(n)
    rng = np.random.default_rng(seed)
    boot_means = np.array([rng.choice(deltas, size=n, replace=True).mean()
                           for _ in range(n_boot)])
    lo, hi = np.quantile(boot_means, [alpha / 2, 1 - alpha / 2])
    return out | {
        "ci": [float(lo), float(hi)],
        "excludes_zero": bool(lo > 0 or hi < 0),
        "p_value": _bootstrap_p(boot_means),
    }


def _bootstrap_p(boot_deltas: np.ndarray) -> float:
    """Two-sided bootstrap p-value: 2 × the fraction of resamples on the null side of 0."""
    frac_le = float(np.mean(boot_deltas <= 0))
    frac_ge = float(np.mean(boot_deltas >= 0))
    return float(min(1.0, 2.0 * min(frac_le, frac_ge)))


def _r2(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot else float("nan")


def bootstrap_r2_ci(y_true, y_pred, *, groups=None, n_boot: int = 2000, alpha: float = 0.05,
                    seed: int = 0) -> dict:
    """CI on a single model's R² by resampling TEST SEQUENCES (for pooled datasets like yeast,
    where per-group counts are too small for per-group R²), or whole DESIGN FAMILIES when
    `groups` gives each row's family."""
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    n = len(y_true)
    units = None if groups is None else _units(groups, n)
    out = {"r2": _r2(y_true, y_pred), "n": int(n),
           "unit": "sequence" if units is None else "family",
           "n_units": n if units is None else len(units)}
    if units is not None and len(units) < 2:
        return out | _no_inference(len(units))
    rng = np.random.default_rng(seed)
    boot = np.array([_r2(y_true[idx], y_pred[idx])
                     for idx in (_draw(rng, n, units) for _ in range(n_boot))])
    lo, hi = np.nanquantile(boot, [alpha / 2, 1 - alpha / 2])
    return out | {"ci": [float(lo), float(hi)]}


def paired_bootstrap_r2(y_true, pred_candidate, pred_baseline, *, groups=None, n_boot: int = 2000,
                        alpha: float = 0.05, seed: int = 0) -> dict:
    """Paired CI on ΔR² (candidate - baseline) by resampling the SAME test rows for both models.

    The unit is one test sequence, unless `groups` labels each row's design family — then whole
    families are resampled (a cluster bootstrap), because a 4-member family is one observation
    drawn four times, not four independent observations. The point estimate is identical either
    way; what changes is how wide the interval honestly is."""
    y_true = np.asarray(y_true, float)
    pc = np.asarray(pred_candidate, float)
    pb = np.asarray(pred_baseline, float)
    n = len(y_true)
    units = None if groups is None else _units(groups, n)
    out = {"mean_delta": float(_r2(y_true, pc) - _r2(y_true, pb)), "n": int(n),
           "unit": "sequence" if units is None else "family",
           "n_units": n if units is None else len(units)}
    if units is not None and len(units) < 2:
        return out | _no_inference(len(units))
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        idx = _draw(rng, n, units)
        deltas.append(_r2(y_true[idx], pc[idx]) - _r2(y_true[idx], pb[idx]))
    deltas = np.array(deltas)
    lo, hi = np.nanquantile(deltas, [alpha / 2, 1 - alpha / 2])
    return out | {"ci": [float(lo), float(hi)],
                  "excludes_zero": bool(lo > 0 or hi < 0),
                  "p_value": _bootstrap_p(deltas[~np.isnan(deltas)])}
