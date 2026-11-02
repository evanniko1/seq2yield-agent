"""ML-01b: the PER-SERIES half of the model-init seed defect (E. coli, the primary benchmark).

ML-01 shipped for the pooled path and was written as if it were global. The post-close audit found
that `experiments/runner.py` still used `spec.seed` for the subsample, the global RNG and the
estimator, so an E. coli verdict was one draw from a distribution nobody had sampled, and the
stored registry held one R² per (series, size, model) from a single baseline seed — so even a
multi-seed candidate had nothing seed-matched to compare against.

These tests pin: the runner's seed loop and its fixed subsample, the (series x seed) estimator and
its exact reduction to the single-seed one, the refusal to pair a multi-seed candidate against a
single-seed registry, and the routing that decides which estimator a comparison gets.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seq2yield.experiments import compare as C  # noqa: E402
from seq2yield.experiments import runner as R  # noqa: E402
from seq2yield.experiments.run_spec import AcceptancePolicy, RunSpec, validate_runspec  # noqa: E402
from seq2yield.statistics.bootstrap import (  # noqa: E402
    paired_bootstrap_ci,
    paired_series_multiseed,
)

POL = AcceptancePolicy(min_delta_r2=0.02, baseline_run_id="x")


def _spec(**kw) -> RunSpec:
    base = dict(run_id="t", model_family="rf", dataset="ecoli")
    return RunSpec(**(base | kw))


# ------------------------------------------------------------------ the runner varies the seed
def test_the_runner_reports_the_seeds_it_will_fit_at():
    assert R.model_seeds(_spec(model_seeds=[3, 1, 2, 1])) == [1, 2, 3]


def test_an_empty_model_seeds_falls_back_to_the_run_seed():
    """A hand-written or replayed older spec has no `model_seeds`; it must still run, at one seed,
    and say so — not silently fit at [1, 2, 3] it never asked for."""
    assert R.model_seeds(_spec(model_seeds=[], seed=7)) == [7]


def test_per_series_rows_carry_the_seed_and_the_subsample_is_held_fixed(require_data, tmp_path):
    """The defining property of the fix: one row per (series, size, seed), and across those seeds
    the TRAINING ROWS are identical — only initialisation differs. Varying the draw as well would
    confound 'the model landed elsewhere' with 'it saw different rows', and sampling is a separate
    intervention with its own control."""
    require_data("ecoli")
    spec = _spec(model_family="rf", model_seeds=[1, 2], train_sizes=[250], iterations=[1],
                 n_series=2, series_selection="prefix")
    out = R.run_runspec(spec)
    m = out["metrics"]
    assert "model_seed" in m.columns
    assert sorted(m["model_seed"].unique().tolist()) == [1, 2]
    assert len(m) == 2 * 2 * 1                       # 2 series x 2 seeds x 1 size
    # the subsample is drawn once per cell, so n_train matches across seeds...
    per_seed = {s: g.set_index("series")["r2"] for s, g in m.groupby("model_seed")}
    # ...and rf is seeded, so two different init seeds must actually give different numbers
    # (if they did not, the seed loop would be measuring nothing)
    assert not np.allclose(per_seed[1].to_numpy(), per_seed[2].to_numpy())


def test_a_single_seed_run_is_unchanged(require_data):
    """`model_seeds == [spec.seed]` must reproduce the pre-2026-11-02 numbers, or every stored
    per-series result becomes unverifiable.

    The tolerance is 1e-15, not 0, and that is a MEASURED property of the estimators rather than
    slack: `rf` runs with `n_jobs=-1`, so sklearn accumulates tree predictions from several threads
    and the float sum depends on thread scheduling. Re-running the identical spec twice gives
    per-series differences of up to 1.1e-16 with no code change at all (which is also, exactly, the
    residual the ML-02 registry rebuild reported). It sits nine orders of magnitude below
    `bootstrap.ZERO_TOL`, so no inference can see it — but it means "bit-for-bit" is the wrong
    phrase for anything that goes through these models."""
    require_data("ecoli")
    kw = dict(model_family="rf", train_sizes=[250], iterations=[1], n_series=2,
              series_selection="prefix", seed=1)
    a = R.run_runspec(_spec(model_seeds=[1], **kw))["metrics"]
    b = R.run_runspec(_spec(model_seeds=[1, 2], **kw))["metrics"]
    b1 = b[b["model_seed"] == 1].reset_index(drop=True)
    assert a["r2"].to_numpy() == pytest.approx(b1["r2"].to_numpy(), abs=1e-15)


# ------------------------------------------------------------------ per-series R² by seed
def test_a_frame_without_the_column_is_one_seed_of_unknown_identity():
    df = pd.DataFrame({"series": [1, 2], "model": ["rf"] * 2, "train_size": [250] * 2,
                       "r2": [0.3, 0.4]})
    by = R.per_series_r2_by_seed(df, 250, "rf")
    assert list(by) == [-1] and by[-1].tolist() == [0.3, 0.4]


def test_a_ragged_grid_is_intersected_not_averaged_over_different_series():
    """Seed 2 is missing series 3. Averaging each seed over its own series set would compare two
    arms measured on different things while still calling the design paired."""
    df = pd.DataFrame({"series": [1, 2, 3, 1, 2], "model": ["rf"] * 5, "train_size": [250] * 5,
                       "model_seed": [1, 1, 1, 2, 2], "r2": [0.1, 0.2, 0.9, 0.1, 0.2]})
    by = R.per_series_r2_by_seed(df, 250, "rf")
    assert sorted(by) == [1, 2]
    for s in by.values():
        assert s.index.tolist() == [1, 2]


def test_iterations_are_averaged_within_a_seed():
    df = pd.DataFrame({"series": [1, 1, 2, 2], "model": ["rf"] * 4, "train_size": [250] * 4,
                       "model_seed": [1, 1, 1, 1], "iteration": ["i1", "i2", "i1", "i2"],
                       "r2": [0.2, 0.4, 0.6, 0.8]})
    by = R.per_series_r2_by_seed(df, 250, "rf")
    assert by[1].tolist() == pytest.approx([0.3, 0.7])


# ------------------------------------------------------------------ the estimator
@pytest.mark.parametrize("n", [3, 5, 10, 20, 56])
def test_one_seed_reduces_exactly_to_the_single_seed_estimator(n):
    """A generalization that is not EXACTLY a generalization cannot be checked. Every field, not
    just the point estimate: an interval computed two ways is two intervals."""
    rng = np.random.default_rng(11)
    base = rng.normal(0.5, 0.1, n)
    cand = base + rng.normal(0.02, 0.0216, n)
    a = paired_bootstrap_ci(base, cand, seed=3)
    b = paired_series_multiseed({9: pd.Series(base)}, {9: pd.Series(cand)}, seed=3)
    for k in ("mean_delta", "ci", "ci_one_sided", "p_value", "p_method", "p_floor",
              "excludes_zero", "ci_excludes_zero", "n_series"):
        assert a[k] == b[k], f"{k}: {a[k]} != {b[k]}"


def test_a_single_seed_result_says_the_seed_component_is_missing():
    rng = np.random.default_rng(2)
    base = pd.Series(rng.normal(0.5, 0.1, 20))
    out = paired_series_multiseed({1: base}, {1: base + 0.03}, seed=0)
    assert out["n_seeds"] == 1 and out["se_seed"] == 0.0
    assert "not estimable" in out["note"]


def test_the_seed_component_widens_the_interval_when_seeds_disagree():
    """The mechanism of the defect: the series interval shrinks with the series count while seed
    noise does not, so an interval built from series variance alone is too narrow by whatever the
    seed contributed."""
    rng = np.random.default_rng(5)
    series_effect = rng.normal(0.0, 0.02, 56)
    arms_b = {s: pd.Series(0.5 + series_effect + rng.normal(0, 0.005, 56)) for s in (1, 2, 3)}
    # each candidate seed lands on a different overall level: seed noise, not a series effect
    arms_c = {s: arms_b[s] + off for s, off in zip((1, 2, 3), (0.004, 0.022, 0.040))}
    multi = paired_series_multiseed(arms_b, arms_c, seed=0)
    single = paired_series_multiseed({1: arms_b[1]}, {1: arms_c[1]}, seed=0)
    assert multi["se_seed"] > 0
    width = lambda d: d["ci"][1] - d["ci"][0]              # noqa: E731
    assert width(multi) > width(single)
    assert multi["seed_spread"] == pytest.approx(0.036, abs=1e-9)


def test_an_effect_that_is_only_a_lucky_seed_is_not_accepted():
    """The A/A shape: both arms are the same model, so the true delta is 0, and the apparent effect
    is which seed each arm happened to draw. The single-seed interval accepts it; the two-component
    one must not."""
    rng = np.random.default_rng(17)
    truth = 0.5 + rng.normal(0, 0.02, 56)
    # one draw per seed: a per-seed level shift plus small per-series noise
    arm = {s: pd.Series(truth + lvl + rng.normal(0, 0.004, 56))
           for s, lvl in zip(range(1, 7), rng.normal(0, 0.03, 6))}
    lucky = max(arm, key=lambda s: arm[s].mean())
    unlucky = min(arm, key=lambda s: arm[s].mean())
    single = paired_series_multiseed({unlucky: arm[unlucky]}, {lucky: arm[lucky]}, seed=0)
    assert single["excludes_zero"] is True             # the defect, reproduced
    # the same data read as what it is: 3 seeds per arm, disjoint -> unpaired
    half = sorted(arm)[:3], sorted(arm)[3:]
    multi = paired_series_multiseed({s: arm[s] for s in half[0]},
                                    {s: arm[s] for s in half[1]}, seed=0)
    assert multi["seed_design"] == "unpaired" and multi["se_seed"] > 0
    assert multi["excludes_zero"] is False


def test_seed_pairs_disagreeing_in_direction_blocks_significance():
    """Sign disagreement across seeds is a veto, not a term in the variance.

    It takes six seeds to show this, and that is informative: with three, any sign disagreement
    implies a seed spread comparable to the mean, so the interval already covers zero on its own
    (the test below). The veto only does independent work once there are enough seeds for one
    dissenter to be outvoted by the variance — which is the case where "the candidate is better"
    is most tempting and least true, because one run of it was not."""
    rng = np.random.default_rng(3)
    b = {s: pd.Series(0.5 + rng.normal(0, 0.01, 30)) for s in range(1, 7)}
    c = {s: (v + 0.05 if s < 6 else v - 0.0001) for s, v in b.items()}
    out = paired_series_multiseed(b, c, seed=0)
    assert out["ci_excludes_zero"] is True             # the interval alone would say yes
    assert out["seed_signs_agree"] is False
    assert out["excludes_zero"] is False               # ...the seeds veto it
    assert "disagree in direction" in out["note"]


def test_a_large_seed_spread_widens_the_interval_over_zero_by_itself():
    rng = np.random.default_rng(3)
    b = {s: pd.Series(0.5 + rng.normal(0, 0.01, 30)) for s in (1, 2, 3)}
    out = paired_series_multiseed(b, {1: b[1] + 0.05, 2: b[2] - 0.05, 3: b[3] + 0.05}, seed=0)
    assert out["ci"][0] < 0 < out["ci"][1] and out["excludes_zero"] is False


def test_a_ragged_seed_grid_is_refused_rather_than_broadcast():
    with pytest.raises(ValueError, match="same series set"):
        paired_series_multiseed({1: pd.Series([0.1, 0.2])},
                                {1: pd.Series([0.1, 0.2, 0.3])}, seed=0)


def test_below_tolerance_reports_a_real_interval_not_a_zero_width_one():
    """Found while building this: the pooled estimator returned ci=[mean, mean] below tolerance. A
    zero-width interval trips `_decide`'s degeneracy guard and comes back inconclusive, when a
    delta of ~1e-12 with real per-series variability is exactly what TOST was added for — the
    difference is SHOWN to be negligible rather than merely unproven."""
    rng = np.random.default_rng(1)
    b = {s: pd.Series(0.5 + rng.normal(0, 0.01, 30)) for s in (1, 2, 3)}
    c = {s: v + 1e-12 * (1.0 + rng.normal(0, 0.1, 30)) for s, v in b.items()}
    out = paired_series_multiseed(b, c, seed=0)
    assert out["p_method"] == "below_tolerance" and out["excludes_zero"] is False
    assert out["ci"][1] > out["ci"][0]
    status, _reasons, verdict = C._decide(out["mean_delta"], False, out["ci"], POL,
                                          ci_one_sided=out["ci_one_sided"])
    assert (status, verdict) == ("rejected", "negligible")


def test_bit_identical_arms_stay_degenerate_and_inconclusive():
    """The other side of the same coin: with EXACTLY equal arms there is no between-unit
    variability to estimate, so a zero-width interval is correct and fitting inside any margin is
    absence of variability, not evidence of equivalence (the ST-3 degeneracy guard)."""
    rng = np.random.default_rng(1)
    b = {s: pd.Series(0.5 + rng.normal(0, 0.01, 30)) for s in (1, 2, 3)}
    out = paired_series_multiseed(b, b, seed=0)
    assert out["ci"] == [0.0, 0.0]
    status, _r, verdict = C._decide(out["mean_delta"], False, out["ci"], POL,
                                    ci_one_sided=out["ci_one_sided"])
    assert (status, verdict) == ("inconclusive", "unresolved")


# ------------------------------------------------------------------ routing + the registry gate
def test_compare_multiseed_intersects_the_series_set_across_arms_and_seeds():
    b = {1: pd.Series([0.1, 0.2, 0.3], index=[1, 2, 3]),
         2: pd.Series([0.1, 0.2, 0.3], index=[1, 2, 3])}
    c = {1: pd.Series([0.2, 0.3], index=[2, 3]), 2: pd.Series([0.2, 0.3], index=[2, 3])}
    out = C.compare_multiseed(b, c, POL, seed=0)
    assert out["n_series"] == 2


def test_per_size_uses_the_two_component_estimator_only_when_both_arms_have_seeds():
    def frame(model, seeds, r2s):
        rows = []
        for s in seeds:
            for i, v in enumerate(r2s, start=1):
                rows.append({"series": i, "model": model, "train_size": 250,
                             "model_seed": s, "r2": v + 0.001 * s})
        return pd.DataFrame(rows)
    vals = list(np.linspace(0.3, 0.6, 12))
    multi = C.compare_per_size(frame("rf", [1, 2, 3], vals),
                               frame("cnn", [1, 2, 3], [v + 0.03 for v in vals]),
                               [250], "rf", "cnn", POL)
    assert multi[0]["ci_method"] == "t_two_component" and multi[0]["n_seeds"] == 3
    one = C.compare_per_size(frame("rf", [1], vals),
                             frame("cnn", [1, 2, 3], [v + 0.03 for v in vals]),
                             [250], "rf", "cnn", POL)
    assert one[0]["ci_method"] == "t"
    assert any("NOT in this interval" in r for r in one[0]["reasons"])


def test_a_single_seed_registry_is_refused_as_a_control_for_a_multi_seed_run(tmp_path):
    """ML-01b(b). Pairing 3 candidate seeds against a 1-seed registry would put the candidate's
    seed variance in the interval and the baseline's nowhere — and the baseline's is not zero, it
    is unmeasured. Same remedy as ML-02's: refuse, and train an in-run control."""
    from orchestration import execution_harness as EH
    run = tmp_path / "experiments/runs/reg"
    run.mkdir(parents=True)
    pd.DataFrame({"series": [1, 2], "model": ["rf"] * 2, "train_size": [250] * 2,
                  "r2": [0.3, 0.4]}).to_csv(run / "metrics.csv", index=False)
    old = EH.ROOT
    try:
        EH.ROOT = tmp_path
        ok, why = EH._registry_seed_coverage("reg", [1, 2, 3])
        assert ok is False and "no model_seed column" in why
        pd.DataFrame({"series": [1, 2] * 3, "model": ["rf"] * 6, "train_size": [250] * 6,
                      "model_seed": [1, 1, 2, 2, 3, 3], "r2": [0.3, 0.4] * 3}
                     ).to_csv(run / "metrics.csv", index=False)
        assert EH._registry_seed_coverage("reg", [1, 2, 3]) == (True, "")
        ok, why = EH._registry_seed_coverage("reg", [1, 2, 9])
        assert ok is False and "missing model_seed [9]" in why
    finally:
        EH.ROOT = old


def test_a_single_seed_registry_is_still_fine_for_a_single_seed_run(tmp_path):
    """The gate is about the PAIRING, not about disliking the registry: a one-seed candidate has no
    seed component to match, so nothing is refused and the existing registry keeps working."""
    from orchestration import execution_harness as EH
    spec = _spec(model_seeds=[1])
    spec.acceptance_policy.baseline_run_id = "2026-11-02-2026-11-02-full56"
    ok, prov = EH._baseline_choice(spec, spec.acceptance_policy)
    assert prov is not None
    assert "model_seed_coverage" not in (prov.get("violations") or [])


def test_three_seeds_are_still_required_for_a_claim():
    v = validate_runspec(_spec(model_seeds=[1, 2], iterations=[1, 2]), unlocked_tier="tier_1")
    assert any("model_seeds" in e for e in v.errors)


# ------------------------------------------------------------------ found on the way
def test_the_audit_log_cannot_kill_the_run_it_is_recording(tmp_path):
    """Found by the end-to-end smoke of this change, and pre-existing: with a single train size the
    harness's `comparison_train_size` came out as a `numpy.int64`, `json.dumps` raised TypeError
    from inside `audit_log.append`, and the harness died at the moment it was writing its own
    verdict. The log is the trail that makes a run trustable, so it is the last thing that may
    fail. Source fixed too (`int(...)` in the harness); this pins the floor."""
    import json

    import numpy as np

    from orchestration import audit_log
    audit_log.append(tmp_path, "compare", {
        "train_size": np.int64(250), "delta": np.float64(0.0126),
        "ci": np.array([0.0069, 0.0183]), "seeds": {1, 2, 3}, "where": tmp_path,
    })
    rec = json.loads((tmp_path / "audit_log.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert rec["train_size"] == 250 and isinstance(rec["train_size"], int)
    assert rec["delta"] == pytest.approx(0.0126)
    assert rec["ci"] == [0.0069, 0.0183]
    assert rec["seeds"] == [1, 2, 3]


def test_the_harness_no_longer_emits_a_numpy_train_size():
    """The source side of the same bug: a set intersection of Python ints with numpy ints can
    return either type, so the value is cast where it is computed rather than only coerced on the
    way out."""
    import numpy as np
    sizes = {250}
    size = int(sizes.intersection(set(np.array([250, 500]))).pop())
    assert isinstance(size, int)
    src = (ROOT / "src/orchestration/execution_harness.py").read_text(encoding="utf-8")
    assert "size = int(pol.comparison_train_size or max(" in src
