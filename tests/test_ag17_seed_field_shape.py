"""AG-17: `n_seeds` means the same thing on every code path (docs/CONTRACTS.md).

Three places build a comparison record — `bootstrap._two_component` (via `compare._package`),
`compare.compare_per_size`'s single-seed fallback, and `execution_harness`'s per-series fallback —
and `n_seeds` used to be an `int` from the first and a `{candidate, baseline}` dict from the other
two. Nothing broke, because nothing read it for logic; the cost was that a field whose whole purpose
is to answer "was the seed component estimated, and from how many seeds per arm?" could not be read
without first knowing which estimator had produced the record.

These tests exercise each producer and assert one shape, so the three cannot drift apart again —
the same failure mode this project keeps finding (ST-3's accept rule in three files, ST-2's
exact-zero p in three estimators).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seq2yield.experiments import compare as C                      # noqa: E402
from seq2yield.experiments.run_spec import AcceptancePolicy         # noqa: E402
from seq2yield.statistics.bootstrap import (paired_series_multiseed,  # noqa: E402
                                            paired_r2_multiseed)

POL = AcceptancePolicy(track="performance", baseline_model="rf", comparison_train_size=250)


def _frame(model, seeds, vals):
    rows = []
    for s in seeds:
        for i, v in enumerate(vals):
            rows.append({"series": i, "model": model, "train_size": 250,
                         "iteration": "iteration_1", "model_seed": s, "r2": v + 0.001 * s})
    return pd.DataFrame(rows)


def _is_per_arm(n_seeds) -> bool:
    return (isinstance(n_seeds, dict) and set(n_seeds) == {"candidate", "baseline"}
            and all(isinstance(v, int) for v in n_seeds.values()))


# ------------------------------------------------------------------- producer 1: the estimator
def test_the_two_component_estimator_reports_per_arm_counts():
    vals = list(np.linspace(0.3, 0.6, 12))
    b = {s: pd.Series(vals) for s in (1, 2, 3)}
    c = {s: pd.Series([v + 0.03 for v in vals]) for s in (1, 2, 3)}
    out = paired_series_multiseed(b, c, seed=0)
    assert _is_per_arm(out["n_seeds"]), out["n_seeds"]
    assert out["n_seeds"] == {"candidate": 3, "baseline": 3}
    assert out["seeds"] == {"candidate": [1, 2, 3], "baseline": [1, 2, 3]}
    assert out["n_seed_pairs"] == 3, "the useful scalar keeps its own precise name"


def test_the_pooled_estimator_reports_the_same_shape():
    """The pooled and per-series estimators share `_two_component`, and the point of sharing it is
    that they cannot report different field sets for the same concept."""
    rng = np.random.default_rng(0)
    y = rng.normal(0.5, 0.2, 200)
    base = {s: y + rng.normal(0, 0.1, 200) for s in (1, 2)}
    cand = {s: y + rng.normal(0, 0.08, 200) for s in (1, 2)}
    out = paired_r2_multiseed(y, cand, base, seed=0)
    assert _is_per_arm(out["n_seeds"]), out["n_seeds"]
    assert out["seeds"] == {"candidate": [1, 2], "baseline": [1, 2]}


def test_an_unpaired_design_can_report_different_counts_per_arm():
    """The case the int could not express at all. Unequal arms are exactly when you most need to
    know which side is short."""
    vals = list(np.linspace(0.3, 0.6, 12))
    b = {s: pd.Series(vals) for s in (1, 2)}
    c = {s: pd.Series([v + 0.03 for v in vals]) for s in (7, 8, 9)}
    out = paired_series_multiseed(b, c, seed=0)
    assert out["n_seeds"] == {"candidate": 3, "baseline": 2}
    assert out["seeds"] == {"candidate": [7, 8, 9], "baseline": [1, 2]}
    assert out["seed_design"] == "unpaired", (
        "different seed ids per arm is the definition of unpaired, and `seeds` is what lets a "
        "reader verify that rather than trust the label")


# ------------------------------------------------------------ producer 2: compare_per_size
@pytest.mark.parametrize("base_seeds,cand_seeds,expect", [
    ([1, 2, 3], [1, 2, 3], {"candidate": 3, "baseline": 3}),
    ([1], [1, 2, 3], {"candidate": 3, "baseline": 1}),
    ([1, 2, 3], [1], {"candidate": 1, "baseline": 3}),
    ([1], [1], {"candidate": 1, "baseline": 1}),
])
def test_compare_per_size_reports_per_arm_counts_on_both_branches(base_seeds, cand_seeds, expect):
    vals = list(np.linspace(0.3, 0.6, 12))
    out = C.compare_per_size(_frame("rf", base_seeds, vals),
                             _frame("cnn", cand_seeds, [v + 0.03 for v in vals]),
                             [250], "rf", "cnn", POL)[0]
    assert _is_per_arm(out["n_seeds"]), out["n_seeds"]
    assert out["n_seeds"] == expect
    assert out["seeds"] == {"candidate": sorted(cand_seeds), "baseline": sorted(base_seeds)}


def test_ci_method_not_n_seeds_is_what_says_which_estimator_ran():
    """The reason one shape is safe: the record already carries an unambiguous discriminator, so
    `n_seeds` never needed to double as one."""
    vals = list(np.linspace(0.3, 0.6, 12))
    multi = C.compare_per_size(_frame("rf", [1, 2, 3], vals),
                               _frame("cnn", [1, 2, 3], [v + 0.03 for v in vals]),
                               [250], "rf", "cnn", POL)[0]
    single = C.compare_per_size(_frame("rf", [1], vals),
                                _frame("cnn", [1, 2, 3], [v + 0.03 for v in vals]),
                                [250], "rf", "cnn", POL)[0]
    assert multi["ci_method"] == "t_two_component" and multi["seed_design"] == "paired"
    assert single["ci_method"] == "t"
    assert single["seed_design"] == "seed_averaged_single_component"
    # and the fallback carries NO se_seed at all, which is the honest encoding: the component was
    # not estimated, so there is no number for it — absent rather than silently zero
    assert "se_seed" not in single and "se_seed" in multi
    assert any("NOT in this interval" in r for r in single["reasons"]), (
        "the deficiency must stay NAMED in reasons, not left to be inferred from a field")


# ------------------------------------------------------- producer 3: the harness, and the records
def test_the_harness_fallback_uses_the_same_shape():
    """Read as source rather than executed: running it needs a trained model and a registry, and
    the thing under test is the literal shape the branch assigns."""
    src = (ROOT / "src" / "orchestration" / "execution_harness.py").read_text(encoding="utf-8")
    assert 'cmp["n_seeds"] = {"candidate": len(cand_by), "baseline": len(base_by)}' in src
    assert 'cmp["seeds"] = {"candidate": sorted(cand_by), "baseline": sorted(base_by)}' in src


def test_every_committed_run_card_uses_the_per_arm_shape():
    """The records on disk, which is where the inconsistency was actually visible."""
    import json
    checked = 0
    for card in sorted((ROOT / "experiments" / "runs").glob("*/run_card.json")):
        primary = (json.loads(card.read_text(encoding="utf-8")).get("results") or {}).get("primary")
        if not isinstance(primary, dict) or "n_seeds" not in primary:
            continue
        assert _is_per_arm(primary["n_seeds"]), f"{card.parent.name}: {primary['n_seeds']!r}"
        checked += 1
    assert checked, "no run-card carried an n_seeds field — this test would pass vacuously"
