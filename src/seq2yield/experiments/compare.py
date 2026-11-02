"""Candidate-vs-baseline comparison and acceptance decision.

CONDITIONALLY PROTECTED (configs/protected_files.yaml): changes require a formal proposal +
human review, because this encodes how scientific claims are accepted. The per-size verdict +
crossover analysis were added under explicit human authorization (DECISIONS #25).
"""
from __future__ import annotations

import pandas as pd

from ..statistics.bootstrap import paired_bootstrap_ci, paired_series_multiseed
from .run_spec import AcceptancePolicy


def _decide(delta: float, ci_excludes_zero: bool, ci, policy: AcceptancePolicy,
            *, ci_one_sided=None) -> tuple[str, list]:
    """The acceptance rule — single source of truth for one comparison.

    AUDIT ST-3: this used to compare the margin with the POINT estimate and the interval with
    ZERO, so it never tested the hypothesis it was asserting. A true ΔR² of 0.019 — just under the
    0.02 margin — was accepted 42.3% of the time at 10 units, 42.3% at 20 and 40.2% at 56:
    precision bought almost nothing, because more data shrank the interval around a point estimate
    that was still above the margin half the time.

    A margin claim is directional, so it is now tested against the ONE-SIDED lower limit:
    `accepted` iff lower ≥ margin, which measures 1.7% at δ=0.019 and 4.2% at δ exactly 0.02
    (correctly sized at the boundary) while keeping 95.3% power at δ=0.03 over 56 units.

    `margin_verdict` carries the full answer, because "not accepted" collapses four different
    situations that the three-status vocabulary cannot distinguish:

      superior_by_margin          lower ≥ +margin                      -> accepted
      inferior_by_margin          upper ≤ −margin                      -> rejected
      superior_but_below_margin   beats 0 AND upper < +margin: better, and the gap is SHOWN to be
                                  under the bar — a conclusive negative, not an unresolved one
      inferior_but_below_margin   worse than 0 but by less than the margin
      negligible                  both one-sided limits inside ±margin and 0 not excluded: TOST,
                                  equivalence SHOWN rather than assumed from a failure to reject
      superior_margin_unresolved  beats 0, but the interval still reaches past the margin
      unresolved                  spans 0 and too wide to call either way

    The TOST branch uses the two one-sided limits (the textbook construction, each test at alpha)
    rather than the two-sided interval, which measures correctly sized — 5.2% false equivalence at
    a true delta exactly equal to the margin — where the two-sided version was conservative (2.6%)
    and much less powerful (equivalence at delta=0 shown 72% vs 49% of the time at 10 units).

    `superior_margin_unresolved` used to be reported as `rejected` ("significant but below the
    practical threshold"), which reads as "the candidate failed" when the truth was "the candidate
    is better and we cannot say by how much". That is the one status this change moves.
    """
    reasons = []
    margin = policy.min_delta_r2
    lo1, hi1 = (ci_one_sided if ci_one_sided else (None, None))
    lo2, hi2 = (ci[0], ci[1]) if ci and ci[0] is not None else (None, None)
    # fall back to the two-sided limits when a caller has no one-sided ones (legacy records):
    # conservative, and it never turns a non-accept into an accept
    lo = lo1 if lo1 is not None else lo2
    hi = hi1 if hi1 is not None else hi2
    import math
    usable = lo is not None and not (math.isnan(lo) or math.isnan(hi))

    if not usable:
        return "inconclusive", ["no usable interval: nothing is established"], "unresolved"
    # A margin claim does not override ST-2's floor: `ci_excludes_zero` is itself the conjunction
    # of a calibrated interval and an exact test, and at 5 or fewer units that test CANNOT reject
    # however large the effect. Without this, the one-sided limit would clear the margin on 5 units
    # and quietly re-grant what ST-2 established is unobtainable.
    if lo >= margin and not ci_excludes_zero:
        return "inconclusive", [f"one-sided lower limit {lo:+.4f} ≥ margin {margin}, but the "
                                "comparison is not distinguishable from zero at this unit count "
                                "(see p_floor): nothing is established"], "unresolved"
    if lo >= margin:
        return "accepted", reasons, "superior_by_margin"
    if hi <= -margin and ci_excludes_zero:
        return "rejected", [f"one-sided upper limit {hi:+.4f} ≤ −margin {margin}: "
                            "significantly worse by the margin"], "inferior_by_margin"
    # Inside the equivalence region. The degeneracy guard matters: a zero-width interval (every
    # unit value identical, e.g. an A/A against itself) fits inside ANY margin, but between-unit
    # variance is then unestimable, so that is absence of variability rather than evidence of
    # equivalence — the same reasoning that makes the bootstrap refuse a 1-unit CI.
    below_margin = hi > lo and -margin < lo and hi < margin
    if below_margin and ci_excludes_zero:
        side = "better" if delta > 0 else "worse"
        verdict = ("superior_but_below_margin" if delta > 0 else "inferior_but_below_margin")
        return "rejected", [f"{side} than the baseline (ΔR²={delta:+.4f}, 0 excluded) but the gap "
                            f"is established to lie in [{lo:+.4f}, {hi:+.4f}], inside the ±{margin}"
                            " margin: a conclusive negative, not an unresolved one"], verdict
    if below_margin:
        return "rejected", [f"both one-sided limits lie inside ±{margin} ([{lo:+.4f}, {hi:+.4f}]) "
                            "and 0 is not excluded: difference shown to be negligible "
                            "(TOST)"], "negligible"
    if ci_excludes_zero and delta > 0:
        return "inconclusive", [f"beats the baseline (0 excluded) but the interval still reaches "
                                f"past the margin ([{lo:+.4f}, {hi:+.4f}] vs ±{margin}): superior, "
                                "margin not established"], "superior_margin_unresolved"
    if ci_excludes_zero and delta <= 0:
        return "rejected", [f"significantly worse than the baseline (ΔR²={delta:+.4f}, "
                            f"CI {ci})"], "inferior_margin_unresolved"
    return "inconclusive", [f"interval {ci} spans 0 and is too wide to establish either a "
                            f"±{margin} difference or its absence"], "unresolved"


def _package(boot: dict, base_mean: float, cand_mean: float, policy: AcceptancePolicy) -> dict:
    """Turn an estimator's output + the acceptance rule into the comparison record.

    One packaging function for both the single-seed and the (series × seed) estimator, so the two
    cannot report different field sets for the same comparison."""
    delta = boot["mean_delta"]
    status, reasons, margin_verdict = _decide(delta, boot["excludes_zero"], boot["ci"], policy,
                                              ci_one_sided=boot.get("ci_one_sided"))
    # how the interval and the p-value were computed travels with them: a t-interval + exact
    # sign-flip verdict is not comparable with a percentile one (audit ST-2, the C3 fence), and a
    # unit count whose exact test cannot reach alpha is a limit of the evidence worth reporting
    if boot.get("note"):
        reasons = reasons + [boot["note"]]
    out = {
        "status": status,
        "baseline_mean": float(base_mean),
        "candidate_mean": float(cand_mean),
        "mean_delta": float(delta),
        "paired_bootstrap_ci": boot["ci"],
        "ci_excludes_zero": boot["excludes_zero"],
        "p_value": boot.get("p_value"),
        "n_series": boot.get("n_series", boot.get("n_units")),
        "ci_method": boot.get("ci_method"),
        "p_method": boot.get("p_method"),
        "p_floor": boot.get("p_floor"),
        # ST-3: the verdict ABOUT THE MARGIN, which the three-status vocabulary cannot express.
        # `margin` travels with it because a verdict is meaningless without the threshold it used.
        "ci_one_sided": boot.get("ci_one_sided"),
        "margin": policy.min_delta_r2,
        "margin_verdict": margin_verdict,
        "reasons": reasons,
    }
    # ML-01b: the seed side of the uncertainty, when there was one. `se_unit`/`se_seed` are what
    # make "the interval is wide because of the seed, not the series" readable instead of inferred.
    for key in ("n_seeds", "seed_design", "seed_deltas", "seed_spread", "seed_signs_agree",
                "se_unit", "se_seed", "se_total", "df"):
        if key in boot:
            out[key] = boot[key]
    return out


def _compare_series(base: pd.Series, cand: pd.Series, policy: AcceptancePolicy, seed: int) -> dict:
    boot = paired_bootstrap_ci(base.values, cand.values, seed=seed)
    return _package(boot, base.mean(), cand.mean(), policy)


def _compare_series_multiseed(base_by_seed: dict, cand_by_seed: dict,
                              policy: AcceptancePolicy, seed: int) -> dict:
    boot = paired_series_multiseed(base_by_seed, cand_by_seed, seed=seed)
    base_mean = float(pd.concat(list(base_by_seed.values())).mean())
    cand_mean = float(pd.concat(list(cand_by_seed.values())).mean())
    return _package(boot, base_mean, cand_mean, policy)


def compare(baseline_per_series: pd.Series, candidate_per_series: pd.Series,
            policy: AcceptancePolicy, *, seed: int = 0) -> dict:
    """Decide accepted | rejected | inconclusive for the performance track (one train size)."""
    common = baseline_per_series.index.intersection(candidate_per_series.index)
    return _compare_series(baseline_per_series.loc[common], candidate_per_series.loc[common],
                           policy, seed)


def compare_multiseed(baseline_by_seed: dict, candidate_by_seed: dict,
                      policy: AcceptancePolicy, *, seed: int = 0) -> dict:
    """Same decision over a (series × model-init seed) grid, carrying BOTH variance components.

    AUDIT ML-01b. `compare` above sees one R² per series and therefore cannot know how much of the
    gap is model initialisation. This takes `{seed: per-series R²}` for each arm and reports an
    interval that includes the seed component, so a candidate that only got a luckier draw is not
    accepted. The series set is intersected across every seed of both arms first: a ragged grid
    would average the two arms over different series and silently break the pairing.

    With one seed per arm the estimator reduces to `compare`'s exactly, so this is safe to call on
    single-seed records — what it is NOT safe to do is call it on a multi-seed candidate against a
    single-seed baseline and read the result as if both components were estimated. That case is
    refused upstream (`execution_harness._baseline_choice`), because the baseline's missing seed
    variance is not zero, it is unmeasured."""
    common = None
    for s in list(baseline_by_seed.values()) + list(candidate_by_seed.values()):
        common = s.index if common is None else common.intersection(s.index)
    if common is None or len(common) == 0:
        return _package({"mean_delta": float("nan"), "ci": [float("nan"), float("nan")],
                         "excludes_zero": False, "p_value": 1.0, "n_units": 0,
                         "note": "no series shared by every arm and seed"},
                        float("nan"), float("nan"), policy)
    return _compare_series_multiseed({k: v.loc[common] for k, v in baseline_by_seed.items()},
                                     {k: v.loc[common] for k, v in candidate_by_seed.items()},
                                     policy, seed)


def compare_per_size(base_df: pd.DataFrame, cand_df: pd.DataFrame, sizes,
                     baseline_model: str, candidate_model: str, policy: AcceptancePolicy,
                     *, seed: int = 0) -> list[dict]:
    """A paired verdict at EACH train size (rigorous per-size data-efficiency curve).

    Routes to the (series × seed) estimator when BOTH arms carry ≥2 model-init seeds, and to the
    series-only one otherwise — with the deficiency named in `reasons` rather than left to be
    inferred from a missing field (audit ML-01b)."""
    from .runner import per_series_r2_by_seed
    out = []
    for size in sorted(set(sizes)):
        b_by = per_series_r2_by_seed(base_df, size, baseline_model)
        c_by = per_series_r2_by_seed(cand_df, size, candidate_model)
        if not b_by or not c_by:
            continue
        if len(b_by) > 1 and len(c_by) > 1:
            res = compare_multiseed(b_by, c_by, policy, seed=seed)
        else:
            b = pd.concat(list(b_by.values()), axis=1).mean(axis=1)
            c = pd.concat(list(c_by.values()), axis=1).mean(axis=1)
            common = b.index.intersection(c.index)
            if len(common) == 0:
                continue
            res = _compare_series(b.loc[common], c.loc[common], policy, seed)
            res["n_seeds"] = {"candidate": len(c_by), "baseline": len(b_by)}
            res["reasons"] = res["reasons"] + [
                f"single-seed arm (candidate {len(c_by)} seed(s), baseline {len(b_by)}): the "
                "model-init variance component is NOT in this interval, so it describes series "
                "variance only (audit ML-01b)"]
        res["train_size"] = int(size)
        out.append(res)
    return out


def heterogeneity_analysis(baseline_per_series: pd.Series, candidate_per_series: pd.Series,
                           *, tie_band: float = 0.005) -> dict:
    """Where does the winner differ ACROSS series? Reports the per-series ΔR² distribution —
    win/loss/tie counts, win rate, and the best/worst series — so 'best on average' vs 'best
    everywhere' is visible instead of collapsed into the mean (answers Q6 heterogeneity)."""
    common = baseline_per_series.index.intersection(candidate_per_series.index)
    if len(common) == 0:
        return {}
    delta = (candidate_per_series.loc[common] - baseline_per_series.loc[common]).sort_values()
    wins = int((delta > tie_band).sum())
    losses = int((delta < -tie_band).sum())
    ties = int(len(delta) - wins - losses)
    return {
        "n_series": int(len(delta)),
        "candidate_wins": wins, "candidate_losses": losses, "ties": ties,
        "win_rate": round(wins / len(delta), 3),
        "tie_band": tie_band,
        "delta_min": round(float(delta.min()), 4),
        "delta_median": round(float(delta.median()), 4),
        "delta_max": round(float(delta.max()), 4),
        "best_series": {"series": int(delta.index[-1]), "delta": round(float(delta.iloc[-1]), 4)},
        "worst_series": {"series": int(delta.index[0]), "delta": round(float(delta.iloc[0]), 4)},
    }


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
