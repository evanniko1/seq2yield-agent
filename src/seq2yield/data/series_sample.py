"""How a bounded subset of mutational series is chosen (audit SA-2).

Its own module, and not a function on `experiments/runner.py`, for two reasons. It decides WHICH
SERIES a stored artifact contains, so it is value-deciding and belongs in the numerics fingerprint;
but `scripts/reproduce_baselines.py` also needs it, and `provenance.artifact_scope` derives a
registry's provenance scope from its builder's import graph — so importing it from `runner` pulled
the whole experiment-execution path into the registry's scope, and every unrelated change to
`runner.py` would then have invalidated a registry whose stored numbers were untouched. That is the
cry-wolf failure the scope was drawn narrowly to avoid (DECISIONS #64/#65).

Three callers share this: `runner.resolve_series`, `experiments/tournament`, and the registry
builder. Each used to carry its own `[:n_series]`.
"""
from __future__ import annotations

import numpy as np

from .cleaning import SERIES_COL
from .loaders import load_split_csv

SERIES_SELECTIONS = ("random", "stratified", "prefix")


def select_series(all_series, n: int | None, *, selection: str = "random", seed: int = 1,
                  sizes: dict | None = None, n_strata: int = 4) -> list[int]:
    """Choose `n` of `all_series`, by a named and reproducible rule.

    Every claim this project has made about E. coli rests on the SAME sorted prefix: the fast probe
    took series[:5], the full run series[:10] and the revisit series[:20], each nested in the next.
    Two things are wrong with that. The subset is not a sample, so no sampling distribution licenses
    generalizing from it to the 56 — and measured on this repo's own registry, CNN−RF shrinks from
    +0.0310 on the first 10 to +0.0210 on all 56 at size 500 and +0.0321 → +0.0228 at 2000.

    The adversarial pass narrowed the mechanism, and the follow-up measurement narrowed it further:
    there is no trend of the delta with series index (Spearman −0.025, p=0.85), and over 2,000
    random k-series subsets the subset delta is UNBIASED (k=10 mean +0.0128 vs the all-56 +0.0126 at
    size 2000, sd 0.0072). So the prefix is not systematically inflated — it is one arbitrary draw,
    and the real costs are that its uncertainty is uncharacterizable and that a 10-series subset
    reproduces an all-56 `accepted` only 17-23% of the time. The nesting is also what makes ST-4's
    optional stopping possible.

    - `random`     seeded uniform sample without replacement. The default, because it is the only
                   option under which a subset estimate has a sampling distribution at all.
    - `stratified` seeded, proportionally allocated across `n_strata` quantiles of the series'
                   WORKING-SET SIZE (`sizes`). Stratifying on a DESIGN property is the point: the
                   audit row suggested stratifying on baseline R², which is an outcome measured on
                   the same held-out sets the run then evaluates on — that is ST-10's defect
                   (selection on held-out R², evaluation on the same sets), so it is not offered.
    - `prefix`     the historical first-N. Kept so a stored bounded reproduction can be rebuilt,
                   and refused as the basis of a claim by `claim_registry.record`.
    """
    ids = list(all_series)
    if selection not in SERIES_SELECTIONS:
        raise ValueError(f"unknown series_selection '{selection}' (have {SERIES_SELECTIONS})")
    if not n or n >= len(ids):
        return ids
    if selection == "prefix":
        return ids[:n]
    rng = np.random.default_rng(seed)
    if selection == "random":
        return sorted(int(x) for x in rng.choice(ids, size=n, replace=False))
    if not sizes:
        raise ValueError("series_selection='stratified' needs per-series working-set sizes; "
                         "pass sizes={series_id: n_rows}")
    order = sorted(ids, key=lambda s: (sizes.get(s, 0), s))
    groups = [g for g in np.array_split(np.asarray(order), min(n_strata, len(order))) if len(g)]
    # proportional allocation, then top up the shortfall from the largest strata so the total is
    # exactly n whatever the rounding did
    take = [max(1, int(round(n * len(g) / len(order)))) for g in groups]
    while sum(take) != n:
        i = int(np.argmax([len(g) - t for g, t in zip(groups, take)])) if sum(take) < n \
            else int(np.argmax(take))
        take[i] += 1 if sum(take) < n else -1
    picked: list[int] = []
    for g, t in zip(groups, take):
        t = min(t, len(g))
        picked += [int(x) for x in rng.choice(g, size=t, replace=False)]
    return sorted(picked)


def series_sizes(splits: dict, iteration: str) -> dict:
    """Per-series working-set row counts for the stratified selection (a design property, never a
    model outcome). `iteration` is a manifest key such as `iteration_1`."""
    work = load_split_csv(splits["iterations"][iteration]["working_set"]["path"])
    return {int(k): int(v) for k, v in work[SERIES_COL].value_counts().items()}
