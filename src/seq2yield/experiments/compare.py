"""Candidate-vs-baseline comparison and acceptance decision.

CONDITIONALLY PROTECTED (configs/protected_files.yaml): changes require a formal proposal +
human review, because this encodes how scientific claims are accepted. The per-size verdict +
crossover analysis were added under explicit human authorization (DECISIONS #25).
"""
from __future__ import annotations

import pandas as pd

from ..statistics.bootstrap import paired_bootstrap_ci
from .run_spec import AcceptancePolicy


def _decide(delta: float, ci_excludes_zero: bool, ci, policy: AcceptancePolicy) -> tuple[str, list]:
    """The acceptance rule — single source of truth for one comparison."""
    reasons = []
    meets_delta = delta >= policy.min_delta_r2
    if not meets_delta:
        reasons.append(f"mean ΔR²={delta:.4f} < min_delta_r2={policy.min_delta_r2}")
    if policy.bootstrap_ci_must_exclude_zero and not ci_excludes_zero:
        reasons.append(f"bootstrap CI {ci} includes 0")

    if meets_delta and (ci_excludes_zero or not policy.bootstrap_ci_must_exclude_zero):
        status = "accepted"
    elif delta <= 0 and ci_excludes_zero:
        status = "rejected"           # candidate is significantly worse
    elif not meets_delta and ci_excludes_zero and delta > 0:
        status = "rejected"           # significant but below the practical threshold
    else:
        status = "inconclusive"       # CI spans zero / underpowered
    return status, reasons


def _compare_series(base: pd.Series, cand: pd.Series, policy: AcceptancePolicy, seed: int) -> dict:
    boot = paired_bootstrap_ci(base.values, cand.values, seed=seed)
    delta = boot["mean_delta"]
    status, reasons = _decide(delta, boot["excludes_zero"], boot["ci"], policy)
    return {
        "status": status,
        "baseline_mean": float(base.mean()),
        "candidate_mean": float(cand.mean()),
        "mean_delta": float(delta),
        "paired_bootstrap_ci": boot["ci"],
        "ci_excludes_zero": boot["excludes_zero"],
        "n_series": boot["n_series"],
        "reasons": reasons,
    }


def compare(baseline_per_series: pd.Series, candidate_per_series: pd.Series,
            policy: AcceptancePolicy, *, seed: int = 0) -> dict:
    """Decide accepted | rejected | inconclusive for the performance track (one train size)."""
    common = baseline_per_series.index.intersection(candidate_per_series.index)
    return _compare_series(baseline_per_series.loc[common], candidate_per_series.loc[common],
                           policy, seed)


def compare_per_size(base_df: pd.DataFrame, cand_df: pd.DataFrame, sizes,
                     baseline_model: str, candidate_model: str, policy: AcceptancePolicy,
                     *, seed: int = 0) -> list[dict]:
    """A paired-bootstrap verdict at EACH train size (rigorous per-size data-efficiency curve)."""
    from .runner import per_series_r2
    out = []
    for size in sorted(set(sizes)):
        b = per_series_r2(base_df, size, baseline_model)
        c = per_series_r2(cand_df, size, candidate_model)
        common = b.index.intersection(c.index)
        if len(common) == 0:
            continue
        res = _compare_series(b.loc[common], c.loc[common], policy, seed)
        res["train_size"] = int(size)
        out.append(res)
    return out


def crossover_analysis(per_size: list[dict]) -> dict:
    """From per-size verdicts, report where the candidate reaches superiority/parity and the
    overall trend — a statistically-grounded answer to 'at what N does it catch up?'."""
    superior_at = next((p["train_size"] for p in per_size if p["status"] == "accepted"), None)
    # parity = no longer significantly worse (CI includes 0) or delta non-negative
    parity_at = next((p["train_size"] for p in per_size
                      if (not p["ci_excludes_zero"]) or p["mean_delta"] >= 0), None)
    trend = "n/a"
    if len(per_size) >= 2:
        d0, d1 = per_size[0]["mean_delta"], per_size[-1]["mean_delta"]
        if abs(d1 - d0) <= 0.01:
            trend = "flat"
        elif d1 > d0:
            trend = "narrowing" if d1 < 0 else "improving"
        else:
            trend = "widening"
    return {"superior_at": superior_at, "parity_at": parity_at, "trend": trend,
            "deltas_by_size": {p["train_size"]: round(p["mean_delta"], 4) for p in per_size}}
