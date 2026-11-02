"""Family-wise selection test: max-T / step-down permutation over a family of candidates.

WHY THIS EXISTS (audit ST-8). The insight layer's "why are these units hard?" answer was the
covariate with the largest |Pearson r| out of a candidate family, reported whenever |r| >= 0.4 and
with no p-value. Reporting the maximum of a family without correcting for the maximisation is not a
test, and the inflation was measured on this repo's own data (2026-11-02, 4000 permutations per
cell, the real E. coli per-series covariate table with only the difficulty vector permuted):

    P(the old selector reports SOME covariate | H0: difficulty independent of every covariate)
        n = 5  ->  0.870          n = 20 ->  0.210
        n = 8  ->  0.690          n = 56 ->  0.008

With four iid covariates instead of the real (correlated) table the same measurement gives
0.941 / 0.782 / 0.293 / 0.006. Pooled neighbourhood dissections run at n <= 8 units, so under the
null the old selector reported an explanation roughly two times in three. The one hit it did report
on real data (`expr_std`, r=+0.53) was substantially R²'s own denominator (audit SA-6).

WHAT THIS FIXES. `step_down_max_t` is the Westfall-Young free step-down max-T procedure: the
adjusted p-value of the k-th largest statistic is the permutation probability that the MAXIMUM over
the not-yet-rejected candidates reaches it, made monotone across the ordered family. It controls the
family-wise error rate at alpha for an arbitrary candidate family and arbitrary dependence between
candidates, which is what the old "max |r| >= 0.4" rule did not do at any n.

Measured on this procedure, same generator, 4000 trials per cell, 4 candidates, alpha=0.05
(exhaustive enumeration of the n! permutations for n <= 6, 1000 sampled permutations above):

    P(reports SOME covariate | H0)
        n = 5  -> 0.047     n = 12 -> 0.056     n = 30 -> 0.052
        n = 6  -> 0.048     n = 20 -> 0.050     n = 56 -> 0.049
        n = 8  -> 0.051

i.e. nominal at every n, including the n <= 8 the old selector reported at two times in three.

WHAT IT DOES NOT FIX. A valid family-wise p-value bounds how often you report nothing-at-all as
something; it does not make the reported effect size trustworthy at small n. Conditional on
rejecting, the selected |r| is biased upward (the winner's curse), because at small n only a
near-perfect alignment can clear the threshold. Measured (one true rho=0.6 covariate in a family of
four, 3000 trials per cell):

    n    power   median |r| reported   inflation over rho=0.6   P(it is the TRUE covariate)
    5    0.084         0.972                  +0.372                      0.60
    6    0.121         0.937                  +0.337                      0.70
    8    0.204         0.870                  +0.270                      0.80
    12   0.365         0.776                  +0.176                      0.92
    20   0.656         0.673                  +0.073                      0.97
    30   0.881         0.624                  +0.024                      0.99
    56   0.994         0.604                  +0.004                      1.00

So the two floors are measurements, not taste:

* `MIN_UNITS_VALID` = 5. With n <= 4 the exact permutation null has at most 24 arrangements, the
  smallest attainable two-sided p-value is 2/24 = 0.083 > 0.05, and NO data can produce a
  rejection -- a report there would be unconditionally invalid. `max_t_correlation` REFUSES below
  this instead of returning an unreachable p-value.
* `MIN_UNITS_RELIABLE` = 20. The first n in the table with better-than-even power (0.66), with the
  inflation of the reported |r| under +0.10, and with the selected covariate being the true one in
  more than 95% of reports. At n = 8 one report in five names the WRONG covariate even when a real
  one exists, and the magnitude is overstated by +0.27. Below the floor a rejection is still VALID,
  so the result is returned with `reliability="low"` and a caveat carrying these numbers rather
  than suppressed -- what is not quotable there is the effect size, not the finding.

The audit row's "require n ~ 20" is right about the effect size and wrong about validity: with a
family-wise p-value the false-report rate is already <= alpha at n = 5. Without one it is 0.87.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from itertools import permutations

import numpy as np

# Smallest n at which the exact permutation null can attain p <= alpha at all (computed, not tuned).
MIN_UNITS_VALID = 5
# Smallest n at which the measured power exceeds 0.5, the conditional inflation of the reported |r|
# falls below 0.10, and >95% of reports name the true covariate (module docstring's table). Below
# this a rejection is valid but the effect size is not quotable.
MIN_UNITS_RELIABLE = 20
# Above this many units, enumerating n! permutations is not affordable -> sample instead.
_EXACT_MAX_PERM = 20000


def min_units_for_alpha(alpha: float = 0.05) -> int:
    """Smallest number of units at which a two-sided exact permutation test can reach `alpha`.

    The exact null over n units has n! arrangements and a symmetric statistic hits its extreme on
    2 of them, so the smallest attainable p-value is 2/n!. Returns the smallest n with
    2/n! <= alpha (5 for alpha=0.05, 6 for alpha=0.01)."""
    for n in range(2, 32):
        if 2.0 / math.factorial(n) <= alpha:
            return n
    return 32


def step_down_max_t(observed: Mapping[str, float],
                    null_draws: Mapping[str, Sequence[float]],
                    *, alpha: float = 0.05) -> dict:
    """Westfall-Young free step-down max-T over a candidate family.

    `observed[name]` is the candidate's statistic (LARGER = more extreme; pass |r|, not r).
    `null_draws[name]` is that candidate's statistics under the null, as a sequence of B draws.
    Draws must be ALIGNED by index across candidates -- draw b of every candidate comes from the
    same null realisation -- or the dependence between candidates is not carried through and the
    procedure silently reverts to a per-candidate test.

    Returns {ordered, adjusted, p_raw, selected, p_max_t, n_family, n_perm, alpha, any_rejected}.
    `adjusted[name]` is the family-wise adjusted p-value; `selected` is the most extreme candidate
    if its adjusted p-value is <= alpha, else None.
    """
    names = [k for k in observed if k in null_draws]
    if not names:
        return {"ordered": [], "adjusted": {}, "p_raw": {}, "selected": None, "p_max_t": None,
                "n_family": 0, "n_perm": 0, "alpha": alpha, "any_rejected": False}
    order = sorted(names, key=lambda k: float(observed[k]), reverse=True)
    M = np.asarray([np.asarray(null_draws[k], dtype=float) for k in order])   # (m, B)
    b = M.shape[1]
    # successive maxima from the least extreme end: suffix_max[k] = max over candidates k..m-1
    suffix_max = np.maximum.accumulate(M[::-1], axis=0)[::-1]                 # (m, B)
    adjusted, p_raw, running = {}, {}, 0.0
    for k, name in enumerate(order):
        t = float(observed[name])
        p = (1.0 + float(np.sum(suffix_max[k] >= t))) / (b + 1.0)             # +1: never report p=0
        running = max(running, p)                                             # monotone step-down
        adjusted[name] = min(1.0, running)
        p_raw[name] = (1.0 + float(np.sum(M[k] >= t))) / (b + 1.0)
    top = order[0]
    rejected = adjusted[top] <= alpha
    return {"ordered": order, "adjusted": adjusted, "p_raw": p_raw,
            "selected": top if rejected else None,
            "p_max_t": adjusted[top], "n_family": len(order), "n_perm": b,
            "alpha": alpha, "any_rejected": bool(rejected)}


def _permutation_matrix(n: int, n_perm: int, seed: int) -> tuple[np.ndarray, bool]:
    """(perm, exact): a (B, n) index matrix. Enumerates all n! permutations when that is affordable,
    which makes the test exact (and the small-n floor exact) instead of Monte-Carlo."""
    if math.factorial(n) <= min(_EXACT_MAX_PERM, max(n_perm, 1)) or math.factorial(n) <= 720:
        return np.asarray(list(permutations(range(n))), dtype=int), True
    rng = np.random.default_rng(seed)
    # argsort of random keys: one vectorised call instead of n_perm Python-level shuffles
    return np.argsort(rng.random((int(n_perm), n)), axis=1), False


def degenerate_columns(values: Mapping[str, Sequence[float]]) -> dict[str, str]:
    """Candidates that cannot be tested, with the reason: too few finite values, or no variation
    across units (a constant covariate has an undefined correlation, not a correlation of zero)."""
    out: dict[str, str] = {}
    for name, col in values.items():
        x = np.asarray(col, dtype=float)
        finite = np.isfinite(x)
        if finite.sum() < 3:
            out[name] = "fewer than 3 finite values"
        elif float(np.std(x[finite])) == 0.0:
            out[name] = "constant across units"
    return out


def max_t_correlation(y, covariates: Mapping[str, Sequence[float]], *,
                      alpha: float = 0.05, n_perm: int = 10000, seed: int = 0,
                      min_units: int | None = None) -> dict:
    """Max-T permutation test for "which of these covariates tracks `y`?".

    The family is every non-degenerate covariate; the statistic is |Pearson r|; the null permutes
    `y` against the covariate table, so the covariates' own correlation structure is preserved and
    carried into the adjusted p-values. `min_units` defaults to `MIN_UNITS_VALID` -- below it the
    test REFUSES rather than returning an unreachable p-value.

    Returns the `step_down_max_t` dict plus: n, r (signed, for the selected candidate), reported,
    refusal, reliability ("ok" | "low"), dropped, and `r_by_covariate`.
    """
    floor = MIN_UNITS_VALID if min_units is None else int(min_units)
    y = np.asarray(y, dtype=float)
    n = int(y.size)
    dropped = degenerate_columns(covariates)
    family = {k: np.asarray(v, dtype=float) for k, v in covariates.items() if k not in dropped}
    base = {"n": n, "dropped": dropped, "family": sorted(family), "n_family": len(family),
            "alpha": alpha, "selected": None, "r": None, "p_max_t": None, "reported": False,
            "reliability": "low", "refusal": None, "adjusted": {}, "r_by_covariate": {}}
    if n < floor:
        base["refusal"] = (f"{n} units < {floor}: at n<{min_units_for_alpha(alpha)} the exact "
                           f"permutation null cannot reach p<={alpha}, so no report is valid")
        return base
    if not family:
        base["refusal"] = "no testable covariate (all candidates degenerate)"
        return base
    if not np.isfinite(y).all() or float(np.std(y)) == 0.0:
        base["refusal"] = "difficulty vector is constant or non-finite"
        return base

    def _z(x):
        x = np.where(np.isfinite(x), x, np.nanmean(np.where(np.isfinite(x), x, np.nan)))
        s = float(np.std(x))
        return (x - float(np.mean(x))) / (s if s else 1.0)

    names = sorted(family)
    Z = np.column_stack([_z(family[k]) for k in names])        # (n, m)
    zy = _z(y)
    r_obs = (Z.T @ zy) / n                                      # (m,)
    perm, exact = _permutation_matrix(n, n_perm, seed)
    R = np.abs((Z.T @ zy[perm.T]) / n)                          # (m, B)
    res = step_down_max_t({k: abs(float(v)) for k, v in zip(names, r_obs)},
                          {k: R[i] for i, k in enumerate(names)}, alpha=alpha)
    r_signed = dict(zip(names, (round(float(v), 4) for v in r_obs)))
    out = {**base, **res, "family": names, "n_family": len(names), "n": n,
           "dropped": dropped, "exact_permutation": bool(exact),
           "r_by_covariate": r_signed}
    if res["selected"]:
        out["r"] = r_signed[res["selected"]]
        out["reported"] = True
    out["reliability"] = "ok" if n >= MIN_UNITS_RELIABLE else "low"
    if out["reliability"] == "low":
        out["effect_size_caveat"] = (
            f"n={n} < {MIN_UNITS_RELIABLE}: conditional on a rejection the reported |r| is biased "
            f"upward (measured +0.27 at n=8, +0.18 at n=12 for a true rho=0.6) and one report in "
            f"five at n=8 names the wrong covariate; the p-value is valid, the magnitude is not")
    return out
