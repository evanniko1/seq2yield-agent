"""Paired inference over per-unit R² deltas (docs/CONTRACTS.md §9, statistics autonomy).

Compares a candidate against a baseline using the paired per-unit differences in R²
(candidate - baseline) and reports an interval on the mean delta plus a test against zero.

THIS WAS NOT CONSERVATIVE, WHICH IS WHAT THE DOCSTRING USED TO CLAIM (audit ST-2). The percentile
bootstrap is liberal at the unit counts this project actually has. Measured at the repo's own noise
scale (per-series cnn−rf sd 0.0216), 1000 trials, nominal 5%, P(the interval excludes 0) under an
exactly-null comparison:

    units                  5      10      20      56
    percentile         15.7%    9.5%    6.6%    6.1%     <- shipped until 2026-11-02
    BCa                16.4%    9.8%    6.2%    6.0%
    t                   5.2%    5.5%    4.2%    5.3%
    this module now     0.0%*   5.1%    4.1%    4.8%     <- t-interval AND exact sign-flip

BCa — the fix the audit suggested first — measurably does not help, and under heavy tails (t₃) it is
the worst of the three (19.3% at n=5, 9.0% at n=56 vs the t-interval's 3.3%/3.8%). So:

- the INTERVAL is a one-sample t-interval on the per-unit values. It needs no resampling, so it is
  also exactly reproducible rather than seed-dependent;
- the P-VALUE is an exact sign-flip permutation test (every one of the 2ⁿ sign patterns enumerated
  for n ≤ 14, Monte Carlo above that), which tests the right null for a paired design — that the
  per-unit differences are symmetric about zero — and makes no normality assumption;
- `excludes_zero` requires BOTH. (*) is why: under enumeration the observed sign pattern and its
  negation always tie with the observation, so the smallest p an exact test can return is
  3/(2ⁿ+1) — 0.091 at n=5, 0.046 at n=6. A 5-unit comparison therefore CANNOT be resolved at
  α=0.05, while the t-interval reaches its nominal 5% there only by assuming a normality that 5
  points cannot evidence. The honest verdict at 5 units is "not resolvable"; `p_floor` carries it.

What this does NOT fix: the sign-flip test assumes the per-unit differences are symmetric about
their centre, and under a strongly skewed delta distribution (standardized Exp(1)) the H0 rate is
still 9.1% at n=10 and 6.5% at n=56. That is a residual, not a solved problem — it is simply much
smaller than the 14.5%/6.4% the percentile method gave on the same draws. Under normal and heavy
(t₃) deltas the rates are 4.1–5.1%.
- `p_value` is floored at 1/(N+1) and never returns exactly 0 (it used to do so in 6.9–9.2% of
  exactly-null comparisons at n=5, and all 8 registry claims carrying a p-value show p=0.0).
- a mean delta at or below `atol` (default 1e-9, far above float noise and far below any real
  effect) is reported as no effect. An A/A whose per-unit deltas were float ULPs — max 1.1e-16 —
  was being called a significant regression, because the residuals were consistently signed.

`method="percentile"` reproduces the old behaviour bit for bit, including its p-value and its
`excludes_zero`, so every pre-2026-11-02 run-card can still be regenerated. The method used is
returned as `ci_method` / `p_method` and recorded alongside the unit, because an interval computed
one way is not comparable with one computed the other (the C3 fence).

THE RESAMPLING UNIT IS THE OTHER HALF OF THE ARGUMENT. A bootstrap CI is only as honest as its unit: the
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

# Below this, a mean delta is numerical noise rather than an effect: R² is O(1), double precision
# gives ~1e-16 relative error, accumulation over a few thousand rows reaches ~1e-12, and no
# scientifically meaningful ΔR² in this project is smaller than 1e-6.
ZERO_TOL = 1e-9
MAX_EXACT_SIGNS = 14          # 2^14 = 16384 patterns enumerated; above this, Monte Carlo
N_SIGN_MC = 4096


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


def _t_interval(values: np.ndarray, alpha: float, *, sided: int = 2) -> tuple[float, float]:
    """One-sample t-interval on the mean of the per-unit values.

    The unit values are the only thing assumed exchangeable, which is exactly the assumption the
    grouped design already forces (ST-7). No resampling, so the interval is deterministic.

    `sided=1` returns the two ONE-sided (1−alpha) limits, each valid on its own. A claim about a
    MARGIN is directional — "better by at least 0.02" — so it is tested against the one-sided lower
    limit, which is correctly sized at the boundary where the two-sided limit is conservative
    (measured: 4.2% vs 1.8% at a true delta exactly equal to the margin, audit ST-3)."""
    from scipy import stats
    n = len(values)
    m = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(n))
    crit = float(stats.t.ppf(1 - (alpha / 2 if sided == 2 else alpha), n - 1))
    return m - crit * se, m + crit * se


def _sign_patterns(n: int, rng) -> tuple[np.ndarray, bool]:
    """All 2ⁿ sign patterns (exact) or a Monte Carlo sample of them."""
    if n <= MAX_EXACT_SIGNS:
        bits = (np.arange(2 ** n)[:, None] >> np.arange(n)) & 1
        return bits.astype(float) * 2.0 - 1.0, True
    return rng.choice([-1.0, 1.0], size=(N_SIGN_MC, n)), False


def _sign_flip_p(values: np.ndarray, rng) -> dict:
    """Two-sided sign-flip permutation test of H0: the per-unit differences are symmetric about 0.

    p is (#{|mean of a flipped sample| >= |observed|} + 1) / (N + 1) — the +1 on both sides is the
    floor that stops a p-value from being reported as exactly 0 when no null sample happened to be
    as extreme as the observation. `p_floor` is the smallest value this test can return, which for
    an enumerated test at small n is a hard limit on what any claim can establish."""
    n = len(values)
    signs, exact = _sign_patterns(n, rng)
    observed = abs(float(values.mean()))
    null = np.abs(signs @ values) / n
    # The identity pattern and its negation reproduce the observed statistic, but by a different
    # summation order, so they can land an ULP below it. The tolerance must therefore be RELATIVE
    # to the statistic: an absolute 1e-18 silently dropped both of them and returned a p BELOW
    # what the test can attain, which at 5 units is the difference between "significant" and
    # "cannot be resolved".
    ge = int((null >= observed * (1.0 - 1e-9) - 1e-18).sum())
    n_patterns = signs.shape[0]
    # smallest ATTAINABLE p, not 1/(N+1): under enumeration those two patterns always tie, so no
    # exact test can report less than 3/(2ⁿ+1)
    floor = (3.0 if exact else 1.0) / (n_patterns + 1.0)
    return {"p_value": max((ge + 1.0) / (n_patterns + 1.0), floor),
            "p_method": "sign_flip_exact" if exact else "sign_flip_mc",
            "p_floor": floor,
            "n_patterns": int(n_patterns)}


def _decide_from_units(values: np.ndarray, *, alpha: float, atol: float, rng) -> dict:
    """The default inference path: t-interval + exact sign-flip p + zero tolerance.

    `values` is one number per resampling unit whose mean IS the statistic of interest, so this
    one function serves the per-series deltas and the linearized pooled ΔR² alike."""
    mean = float(values.mean())
    lo, hi = _t_interval(values, alpha)
    # the one-sided limits travel too: a margin claim is directional and must be tested against
    # them, not against the two-sided interval or the point estimate (audit ST-3)
    lo1, hi1 = _t_interval(values, alpha, sided=1)
    out = {"ci": [lo, hi], "ci_one_sided": [lo1, hi1], "ci_method": "t", "alpha": alpha}
    if abs(mean) <= atol:
        return out | {"excludes_zero": False, "p_value": 1.0, "p_method": "below_tolerance",
                      "p_floor": 1.0,
                      "note": f"|mean delta| = {abs(mean):.3g} is at or below the numerical "
                              f"tolerance {atol:g}: reported as no effect, not as a result"}
    out |= _sign_flip_p(values, rng)
    ci_excl = bool(lo > 0 or hi < 0)
    out["ci_excludes_zero"] = ci_excl
    out["excludes_zero"] = bool(ci_excl and out["p_value"] <= alpha)
    if out["p_floor"] > alpha:
        out["note"] = (f"{len(values)} units: the exact test's smallest attainable two-sided p is "
                       f"{out['p_floor']:.3f} > alpha={alpha}, so no comparison at this unit count "
                       "can be resolved — this is a limit of the evidence, not of the method")
    elif ci_excl and not out["excludes_zero"]:
        out["note"] = ("the t-interval excludes 0 but the exact test does not "
                       f"(p={out['p_value']:.4f}): treated as unresolved")
    return out


def paired_bootstrap_ci(baseline_per_series, candidate_per_series, *,
                        n_boot: int = 10000, alpha: float = 0.05, seed: int = 0,
                        method: str = "t", atol: float = ZERO_TOL) -> dict:
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
    if method != "percentile":
        return out | _decide_from_units(deltas, alpha=alpha, atol=atol, rng=rng)
    boot_means = np.array([rng.choice(deltas, size=n, replace=True).mean()
                           for _ in range(n_boot)])
    lo, hi = np.quantile(boot_means, [alpha / 2, 1 - alpha / 2])
    lo1, hi1 = np.quantile(boot_means, [alpha, 1 - alpha])
    return out | {
        "ci": [float(lo), float(hi)],
        "ci_one_sided": [float(lo1), float(hi1)],
        "ci_method": "percentile",
        "p_method": "bootstrap_percentile",
        "excludes_zero": bool(lo > 0 or hi < 0),
        "p_value": _bootstrap_p(boot_means),
    }


def _bootstrap_p(boot_deltas: np.ndarray) -> float:
    """Two-sided bootstrap p-value: 2 × the fraction of resamples on the null side of 0.

    LEGACY, reachable only via `method="percentile"`, and kept only so pre-2026-11-02 numbers
    reproduce. It returns exactly 0 whenever no resample lands on the null side — which happened in
    6.9–9.2% of exactly-null comparisons at 5 units, and is why every registry claim carrying a
    p-value shows p=0.0 (audit ST-2). The default path uses `_sign_flip_p`, which has a floor."""
    frac_le = float(np.mean(boot_deltas <= 0))
    frac_ge = float(np.mean(boot_deltas >= 0))
    return float(min(1.0, 2.0 * min(frac_le, frac_ge)))


def _r2(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot else float("nan")


def _ss_tot(y_true: np.ndarray) -> float:
    return float(np.sum((y_true - y_true.mean()) ** 2))


def _unit_sums(per_row: np.ndarray, units: list[np.ndarray] | None) -> np.ndarray:
    """Sum `per_row` within each resampling unit (identity when the unit is one row)."""
    if units is None:
        return per_row
    return np.array([float(per_row[idx].sum()) for idx in units])


def _linearize(per_row: np.ndarray, units, offset: float = 0.0) -> np.ndarray:
    """Per-unit values whose MEAN is the statistic, holding SS_tot at its observed value.

    R² and ΔR² are both affine in the per-row squared errors once the denominator is fixed, so a
    statistic over the pooled test set becomes a mean over units — which is what lets the t-interval
    and the sign-flip test (both defined on a mean of exchangeable values) apply to the pooled case
    at all. Fixing SS_tot is the paired, conditional-on-this-test-set reading: SS_tot is a property
    of the held-out set, not of either model. The observed point estimate is unchanged by this."""
    sums = _unit_sums(per_row, units)
    u = len(sums)
    return offset + u * sums


def bootstrap_r2_ci(y_true, y_pred, *, groups=None, n_boot: int = 2000, alpha: float = 0.05,
                    seed: int = 0, method: str = "t") -> dict:
    """CI on a single model's R² by resampling TEST SEQUENCES (for pooled datasets like yeast,
    where per-group counts are too small for per-group R²), or whole DESIGN FAMILIES when
    `groups` gives each row's family.

    Descriptive: there is no paired null here, so no p-value is reported — the interval's position
    relative to 0 answers "does this model beat the training mean" and the caller reads it."""
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
    if method != "percentile":
        ss = _ss_tot(y_true)
        if not ss:
            return out | _no_inference(out["n_units"])
        values = _linearize(-((y_true - y_pred) ** 2) / ss, units, offset=1.0)
        lo, hi = _t_interval(values, alpha)
        return out | {"ci": [lo, hi], "ci_method": "t", "alpha": alpha}
    boot = np.array([_r2(y_true[idx], y_pred[idx])
                     for idx in (_draw(rng, n, units) for _ in range(n_boot))])
    lo, hi = np.nanquantile(boot, [alpha / 2, 1 - alpha / 2])
    return out | {"ci": [float(lo), float(hi)], "ci_method": "percentile"}


def _two_component(V: np.ndarray, *, n_units: int, cs: list, bs: list, paired: bool,
                   alpha: float, atol: float, rng, arm_c=None, arm_b=None) -> dict:
    """The two-component (random-effects) interval, shared by every multiseed estimator.

    `V` is (n_seed_pairs, n_units): one row per seed pair, one column per resampling unit, each
    entry a per-unit value whose MEAN over units is that pair's ΔR². The pooled path builds those
    entries by linearizing per-row squared errors (`_linearize`); the per-series path's entries are
    already per-series ΔR². Everything after that is identical, so it lives here once — two sibling
    estimators that drift apart are the failure mode this project keeps finding in its own code.

        unit component   var(mean over seeds of V[.][u]) / U
        seed component   paired: var of the per-pair deltas / S
                         unpaired: each arm's own seed variance / that arm's count, summed
        se_total         sqrt(unit + seed)
        df               Welch-Satterthwaite across the two components
    """
    from scipy import stats
    mean = float(V.mean())
    seed_deltas = V.mean(axis=1)
    n_pairs = V.shape[0]
    out = {"mean_delta": mean, "n_seeds": len(cs), "seeds": {"candidate": cs, "baseline": bs},
           "seed_design": "paired" if paired else "unpaired", "n_seed_pairs": n_pairs,
           "ci_method": "t_two_component",
           "seed_deltas": [float(x) for x in seed_deltas],
           "seed_spread": float(seed_deltas.max() - seed_deltas.min()) if n_pairs > 1 else 0.0,
           "seed_signs_agree": bool(len(set(np.sign(seed_deltas[seed_deltas != 0]))) <= 1)}
    if n_units < 2:
        return out | _no_inference(n_units)

    unit_means = V.mean(axis=0)                  # seed-averaged per-unit values
    # se FIRST, variance from it — not the other way round. sqrt(var/n) and std/sqrt(n) differ in
    # the last ULP, and `_t_interval` uses the latter, so computing the variance first made the
    # single-seed case fail to reduce to `paired_bootstrap_ci` exactly (measured: 1.7e-18 on one
    # limit at 10 units). Harmless numerically, but an estimator that claims to be a strict
    # generalization of another has to BE one, or the claim is untestable.
    se_unit = float(unit_means.std(ddof=1) / np.sqrt(n_units))
    var_unit = se_unit * se_unit
    n_seed_eff = len(cs) if paired else min(len(cs), len(bs))
    if paired and len(cs) > 1:
        var_seed = float(seed_deltas.var(ddof=1) / len(cs))
        df_seed = len(cs) - 1
    elif (not paired) and arm_c is not None and len(cs) > 1 and len(bs) > 1:
        var_seed = float(np.var(arm_c, ddof=1) / len(cs) + np.var(arm_b, ddof=1) / len(bs))
        df_seed = len(cs) + len(bs) - 2
        out["arm_seed_r2"] = {"candidate": [float(x) for x in arm_c],
                              "baseline": [float(x) for x in arm_b]}
    else:
        var_seed, df_seed = 0.0, 0
    if var_seed > 0:
        df = ((var_unit + var_seed) ** 2 /
              (var_unit ** 2 / max(n_units - 1, 1) + var_seed ** 2 / max(df_seed, 1)))
    else:                                        # one seed: the component is not estimable
        df = float(n_units - 1)
        out["note"] = ("ONE model-init seed per arm: the seed variance component is not estimable, "
                       "so this interval describes only unit variance (audit ML-01/ML-01b — two "
                       "runs of the same model differing only in the seed were accepted in 17% of "
                       "A/A pairs on cuperus_2017)")
    se = se_unit if var_seed == 0.0 else float(np.sqrt(var_unit + var_seed))
    out["se_unit"], out["se_seed"], out["se_total"] = se_unit, float(np.sqrt(var_seed)), se
    out["n_seeds_effective"] = int(n_seed_eff)
    df = max(float(df), 1.0)
    out["df"] = df
    out["alpha"] = alpha
    for key, a in (("ci", alpha / 2), ("ci_one_sided", alpha)):
        crit = float(stats.t.ppf(1 - a, df))
        out[key] = [mean - crit * se, mean + crit * se]
    # The interval is reported even below tolerance. A zero-width [mean, mean] reads as "no
    # variability", which `compare._decide`'s degeneracy guard then treats as unestimable and
    # returns inconclusive — but a delta of 1e-12 with a tight interval is precisely established
    # equivalence, which is what TOST is for. So: real interval, no p-value.
    if abs(mean) <= atol:
        return out | {"excludes_zero": False, "ci_excludes_zero": False, "p_value": 1.0,
                      "p_method": "below_tolerance", "p_floor": 1.0,
                      "note": f"|mean delta| = {abs(mean):.3g} is at or below the numerical "
                              f"tolerance {atol:g}: reported as no effect, not as a result"}
    # The exact test stays unit-level (the seed-averaged per-unit values are the exchangeable
    # quantities); the seed component is carried by the interval, and `seed_signs_agree` exposes
    # the case where seed pairs disagree in direction.
    out |= _sign_flip_p(unit_means, rng)
    ci_excl = bool(out["ci"][0] > 0 or out["ci"][1] < 0)
    out["ci_excludes_zero"] = ci_excl
    out["excludes_zero"] = bool(ci_excl and out["p_value"] <= alpha and out["seed_signs_agree"])
    if ci_excl and not out["seed_signs_agree"]:
        out["note"] = (f"seed pairs disagree in direction ({out['seed_deltas']}): the interval "
                       "excludes 0 but the effect is not consistent across model-init seeds")
    elif out["p_floor"] > alpha:
        out["note"] = (f"{n_units} units: the exact test's smallest attainable two-sided p is "
                       f"{out['p_floor']:.3f} > alpha={alpha}, so no comparison at this unit count "
                       "can be resolved — this is a limit of the evidence, not of the method")
    return out


def paired_series_multiseed(baseline_by_seed: dict, candidate_by_seed: dict, *,
                            alpha: float = 0.05, seed: int = 0, atol: float = ZERO_TOL) -> dict:
    """Per-series ΔR² with BOTH variance components: across SERIES and across model-init seeds.

    AUDIT ML-01b — the per-series half of ML-01, which the pooled fix did not reach. `runner.py`
    never varied the model-init seed, so an ecoli verdict was one draw from a distribution nobody
    had sampled, and the stored registry held one R² per (series, size, model) from a single
    baseline seed, so even a multi-seed candidate had nothing seed-matched to compare against.

    Each value of `*_by_seed` is one R² per series, in a consistent series order. With both arms
    fitted at the same init seeds the design is PAIRED (what the harness does: the seed effect
    common to both models cancels). With disjoint seed sets it is UNPAIRED, which is the shape of a
    seed-only A/A — the same model at different seeds, where there is nothing to pair.

    With a single seed per arm this reduces EXACTLY to `paired_bootstrap_ci`'s t-interval and
    sign-flip p-value, bit for bit (pinned by a test), so single-seed records stay comparable.
    """
    cs, bs = sorted(candidate_by_seed), sorted(baseline_by_seed)
    paired = bool(cs) and cs == bs
    pairs = [(s, s) for s in cs] if paired else [(a, b) for a in cs for b in bs]
    lens = {len(np.asarray(v, float)) for v in
            list(candidate_by_seed.values()) + list(baseline_by_seed.values())}
    if len(lens) > 1:
        raise ValueError(f"every seed must report the same series set (got lengths {sorted(lens)})")
    n_units = lens.pop() if lens else 0
    base = {"unit": "series", "n_units": int(n_units), "n_series": int(n_units)}
    if not pairs or n_units == 0:
        return base | {"mean_delta": float("nan"), "n_seeds": len(cs),
                       "seed_design": "paired" if paired else "unpaired"} | _no_inference(n_units)
    V = np.vstack([np.asarray(candidate_by_seed[a], float) - np.asarray(baseline_by_seed[b], float)
                   for a, b in pairs])
    arm_c = arm_b = None
    if not paired:
        arm_c = np.array([float(np.asarray(candidate_by_seed[s], float).mean()) for s in cs])
        arm_b = np.array([float(np.asarray(baseline_by_seed[s], float).mean()) for s in bs])
    return base | _two_component(V, n_units=n_units, cs=cs, bs=bs, paired=paired, alpha=alpha,
                                 atol=atol, rng=np.random.default_rng(seed),
                                 arm_c=arm_c, arm_b=arm_b)


def paired_r2_multiseed(y_true, cand_by_seed: dict, base_by_seed: dict, *, groups=None,
                        alpha: float = 0.05, seed: int = 0, atol: float = ZERO_TOL) -> dict:
    """ΔR² with BOTH variance components: across resampling units AND across model-init seeds.

    AUDIT ML-01. A verdict used to come from one training seed while the interval resampled test
    rows only, so seed variance was read as signal. Measured on this project's own data, two runs of
    the SAME model differing only in the torch seed were called `accepted` in 7 of 30 ordered pairs
    on dream2022 (worst ΔR² +0.0366, CI [+0.024, +0.050]) and 7 of 20 on cuperus_2017 (worst
    +0.0558). Nothing in the repo varied that seed, so nothing could see it.

    The estimator is the standard two-component (random-effects) form on the ST-2 linearization.
    With `v[s][u]` the per-unit value for seed pair s (its mean over u is that pair's ΔR²):

        unit component   var(mean over seeds of v[.][u]) / U
        seed component   var(mean over units of v[s][.]) / S
        se_total         sqrt(unit + seed)        -- independent components
        df               Welch-Satterthwaite across the two

    Both components are needed: the unit component alone is the old, too-narrow interval, and the
    seed component alone would ignore that a per-unit effect can be consistent across the test set.
    `seed_deltas` and `seed_spread` are reported so the reader can see how much of the uncertainty
    is the seed, and `n_seeds` so a single-seed result is identifiable as such.
    """
    y_true = np.asarray(y_true, float)
    cs, bs = sorted(cand_by_seed), sorted(base_by_seed)
    n = len(y_true)
    units = None if groups is None else _units(groups, n)
    n_units = n if units is None else len(units)
    ss = _ss_tot(y_true)
    # PAIRED when both arms were fitted at the same init seeds (what the harness does: pairing
    # removes the seed effect common to both models and gives the tighter interval). UNPAIRED when
    # the seed sets differ — which is exactly the shape of a seed-only A/A diagnostic, where the
    # two arms are the same model at DIFFERENT seeds, so there is nothing to pair.
    paired = bool(cs) and cs == bs
    pairs = [(s, s) for s in cs] if paired else [(a, b) for a in cs for b in bs]
    base = {"unit": "sequence" if units is None else "family", "n_units": int(n_units)}
    if not pairs or not ss:
        return base | {"mean_delta": float("nan"), "n_seeds": len(cs),
                       "seed_design": "paired" if paired else "unpaired"} | _no_inference(n_units)

    V = np.vstack([_linearize(((y_true - np.asarray(base_by_seed[b], float)) ** 2
                               - (y_true - np.asarray(cand_by_seed[a], float)) ** 2) / ss, units)
                   for a, b in pairs])            # (len(pairs), U)
    arm_c = arm_b = None
    if not paired:
        # each arm's own per-seed R², on the same linearization, so the two independent arms'
        # seed variances can be summed
        arm_c = np.array([_linearize(-((y_true - np.asarray(cand_by_seed[s], float)) ** 2) / ss,
                                     units).mean() for s in cs])
        arm_b = np.array([_linearize(-((y_true - np.asarray(base_by_seed[s], float)) ** 2) / ss,
                                     units).mean() for s in bs])
    return base | _two_component(V, n_units=n_units, cs=cs, bs=bs, paired=paired, alpha=alpha,
                                 atol=atol, rng=np.random.default_rng(seed),
                                 arm_c=arm_c, arm_b=arm_b)


def paired_bootstrap_r2(y_true, pred_candidate, pred_baseline, *, groups=None, n_boot: int = 2000,
                        alpha: float = 0.05, seed: int = 0, method: str = "t",
                        atol: float = ZERO_TOL) -> dict:
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
    if method != "percentile":
        ss = _ss_tot(y_true)
        if not ss:
            return out | _no_inference(out["n_units"])
        # paired per-row loss difference: baseline's squared error minus the candidate's, so a
        # positive unit value means the candidate did better on that unit
        values = _linearize(((y_true - pb) ** 2 - (y_true - pc) ** 2) / ss, units)
        return out | _decide_from_units(values, alpha=alpha, atol=atol, rng=rng)
    deltas = []
    for _ in range(n_boot):
        idx = _draw(rng, n, units)
        deltas.append(_r2(y_true[idx], pc[idx]) - _r2(y_true[idx], pb[idx]))
    deltas = np.array(deltas)
    lo, hi = np.nanquantile(deltas, [alpha / 2, 1 - alpha / 2])
    lo1, hi1 = np.nanquantile(deltas, [alpha, 1 - alpha])
    return out | {"ci": [float(lo), float(hi)],
                  "ci_one_sided": [float(lo1), float(hi1)],
                  "ci_method": "percentile",
                  "p_method": "bootstrap_percentile",
                  "excludes_zero": bool(lo > 0 or hi < 0),
                  "p_value": _bootstrap_p(deltas[~np.isnan(deltas)])}
