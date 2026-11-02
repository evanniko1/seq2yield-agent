"""Mixed-effects / random-effects analysis of GROUPED results (the E. coli 56 mutational series).

The benchmark is a grouped design — 56 series, each measured across MC-CV iterations. Bootstrapping
over series gives a CI; a random-effects model answers the flagship Nat Comms question directly:
**is there a universal optimum or genuine per-series heterogeneity?** It decomposes the variance of
a per-series metric into BETWEEN-series (structural) and WITHIN-series (iteration/seed noise), reports
the **intraclass correlation (ICC = between / total)**, and an F-test of H0 "no between-series
variance". High ICC + small p ⇒ series genuinely differ (heterogeneity); ICC ≈ 0 ⇒ a shared optimum.

Primary estimator is a dependency-free unbalanced one-way random-effects ANOVA (exact, standard).
`mixedlm_random_intercept` is an optional REML wrapper (statsmodels) for a full model with fixed
effects (e.g. train_size) — used only if statsmodels is installed.
"""
from __future__ import annotations

import numpy as np
from scipy import stats as sstats


def variance_components(groups, values) -> dict:
    """Unbalanced one-way random-effects variance decomposition of `values` grouped by `groups`.

    Returns between/within/total variance, ICC, and the F-test p-value for H0: between-group
    variance = 0 (i.e. no genuine group heterogeneity)."""
    g = np.asarray(groups)
    y = np.asarray(values, dtype=float)
    keep = np.isfinite(y)
    g, y = g[keep], y[keep]
    labels = np.unique(g)
    k, N = len(labels), len(y)
    if k < 2 or N - k < 1:
        return {"n_groups": int(k), "n_obs": int(N), "icc": None,
                "note": "need ≥2 groups and ≥1 within-group residual df (replication) for an ICC"}
    grand = float(y.mean())
    sizes, means = [], []
    ssw = 0.0
    for lab in labels:
        yi = y[g == lab]
        sizes.append(len(yi))
        means.append(yi.mean())
        ssw += float(np.sum((yi - yi.mean()) ** 2))
    sizes = np.asarray(sizes, float)
    means = np.asarray(means, float)
    ssb = float(np.sum(sizes * (means - grand) ** 2))
    msb, msw = ssb / (k - 1), ssw / (N - k)
    n0 = (N - float(np.sum(sizes ** 2)) / N) / (k - 1)          # effective group size (unbalanced)
    var_within = float(msw)
    var_between = max(0.0, (msb - msw) / n0)
    total = var_between + var_within
    icc = (var_between / total) if total > 0 else 0.0
    # Zero within-group variance makes F undefined, not infinitely significant. Reporting p=0.0
    # there was the same degenerate-significance pattern audit ST-2 removed from the bootstrap:
    # a statistic that cannot be computed must not come back as the strongest possible evidence.
    degenerate = not (msw > 0)
    f_stat = (msb / msw) if not degenerate else float("inf")
    p_raw = float(sstats.f.sf(f_stat, k - 1, N - k)) if not degenerate else 1.0
    # ST-2's rule, which this module was still breaking in its NON-degenerate branch: a p-value of
    # exactly 0 is not a result. `f.sf` returns 0.0 by underflow once F is large enough — measured
    # live on the E. coli registry, where F ≈ 70 over 56 series gave p = 0.0 — and "0" reads as
    # infinite evidence rather than "below what this computation can represent". The 2026-11-02
    # audit pass fixed the msw == 0 case here and missed this one.
    p_floor = float(np.finfo(float).eps)
    underflow = (not degenerate) and p_raw < p_floor
    p_value = max(p_raw, p_floor) if not degenerate else 1.0
    return {"n_groups": int(k), "n_obs": int(N), "grand_mean": round(grand, 4),
            "var_between": round(var_between, 6), "var_within": round(var_within, 6),
            "var_total": round(total, 6), "icc": round(icc, 4),
            "f_stat": round(f_stat, 4),
            # not rounded to 6 dp: that turned every small p into 0.0 in the record as well
            "p_value": p_value, "p_floor": p_floor, "p_underflow": bool(underflow),
            "degenerate": bool(degenerate),
            "heterogeneous": bool(not degenerate and p_value < 0.05 and icc >= 0.1),
            "verdict": ("within-group variance is 0, so the F test is undefined: nothing is "
                        "established either way" if degenerate else
                        "per-series heterogeneity (between-series variance is real)"
                        if (p_value < 0.05 and icc >= 0.1) else
                        "consistent with a shared optimum (no significant between-series variance)")}


def from_metrics(df, *, model: str, train_size: int, metric: str = "r2",
                 group_col: str = "series") -> dict:
    """Variance components of a per-series metric from a runner/registry metrics frame (columns
    include series, model, train_size, r2). Groups = series; replication = the MC-CV iterations."""
    sub = df[(df["model"] == model) & (df["train_size"] == train_size)]
    if sub.empty:
        raise ValueError(f"no rows for model={model}, train_size={train_size}")
    out = variance_components(sub[group_col].to_numpy(), sub[metric].to_numpy())
    out.update({"model": model, "train_size": train_size, "metric": metric})
    return out


def mixedlm_random_intercept(df, value_col: str, group_col: str, fixed_cols=None) -> dict:
    """Optional REML random-intercept fit (statsmodels): value ~ 1 [+ fixed_cols] + (1|group).
    Returns the group-variance / residual / ICC + fixed-effect params. Raises if statsmodels absent."""
    try:
        import statsmodels.formula.api as smf
    except ImportError as e:  # pragma: no cover
        raise ImportError("mixedlm_random_intercept needs statsmodels (`pip install statsmodels`); "
                          "use variance_components() for the dependency-free ICC") from e
    rhs = " + ".join(fixed_cols) if fixed_cols else "1"
    res = smf.mixedlm(f"{value_col} ~ {rhs}", df, groups=df[group_col]).fit(reml=True)
    grp_var = float(res.cov_re.iloc[0, 0])
    resid = float(res.scale)
    total = grp_var + resid
    return {"group_var": round(grp_var, 6), "resid_var": round(resid, 6),
            "icc": round(grp_var / total, 4) if total > 0 else 0.0,
            "params": {k: round(float(v), 4) for k, v in res.fe_params.items()}}
