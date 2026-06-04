"""Candidate-vs-baseline comparison and acceptance decision.

CONDITIONALLY PROTECTED (configs/protected_files.yaml): changes require a formal proposal +
human review, because this encodes how scientific claims are accepted.
"""
from __future__ import annotations

import pandas as pd

from ..statistics.bootstrap import paired_bootstrap_ci
from .run_spec import AcceptancePolicy


def compare(baseline_per_series: pd.Series, candidate_per_series: pd.Series,
            policy: AcceptancePolicy, *, seed: int = 0) -> dict:
    """Decide accepted | rejected | inconclusive for the performance track.

    Pairs by series (intersection of indices), computes mean delta + paired bootstrap CI,
    and applies the acceptance policy.
    """
    common = baseline_per_series.index.intersection(candidate_per_series.index)
    base = baseline_per_series.loc[common]
    cand = candidate_per_series.loc[common]

    boot = paired_bootstrap_ci(base.values, cand.values, seed=seed)
    delta = boot["mean_delta"]

    reasons = []
    meets_delta = delta >= policy.min_delta_r2
    ci_excludes_zero = boot["excludes_zero"]

    if not meets_delta:
        reasons.append(f"mean ΔR²={delta:.4f} < min_delta_r2={policy.min_delta_r2}")
    if policy.bootstrap_ci_must_exclude_zero and not ci_excludes_zero:
        reasons.append(f"bootstrap CI {boot['ci']} includes 0")

    if meets_delta and (ci_excludes_zero or not policy.bootstrap_ci_must_exclude_zero):
        status = "accepted"
    elif delta <= 0 and ci_excludes_zero:
        status = "rejected"           # candidate is significantly worse
    elif not meets_delta and ci_excludes_zero and delta > 0:
        status = "rejected"           # significant but below the practical threshold
    else:
        status = "inconclusive"       # CI spans zero / underpowered

    return {
        "status": status,
        "baseline_mean": float(base.mean()),
        "candidate_mean": float(cand.mean()),
        "mean_delta": float(delta),
        "paired_bootstrap_ci": boot["ci"],
        "ci_excludes_zero": ci_excludes_zero,
        "n_series": boot["n_series"],
        "reasons": reasons,
    }
