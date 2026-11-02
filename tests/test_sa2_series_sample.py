"""SA-2: a series subset must be a SAMPLE, and a sorted prefix is not one.

Every E. coli claim this project has made rests on the same sorted prefix — the fast probe took
series[:5], the full run series[:10], the revisit series[:20], each nested in the next, and
`tournament` carried a third copy of `[:n_series]` that wrote accepted claims directly. Measured on
the registry, CNN−RF shrinks from +0.0310 on the first 10 to +0.0210 on all 56 at size 500.

The adversarial pass narrowed the mechanism and that narrowing is what these tests encode: there is
no trend of the delta with series index, so the prefix is not biased by sorting — it is one
arbitrary subset whose estimate has no sampling distribution at all. So the fix is the SELECTION
RULE, recorded and seeded, plus a claim gate; it is not "use more series".
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seq2yield.experiments import claim_registry  # noqa: E402
from seq2yield.experiments import runner as R  # noqa: E402
from seq2yield.experiments.run_spec import RunSpec, validate_runspec  # noqa: E402

ALL = list(range(1, 57))


def _spec(**kw) -> RunSpec:
    base = dict(run_id="t", model_family="rf", dataset="ecoli")
    return RunSpec(**(base | kw))


# ------------------------------------------------------------------ the selector
def test_prefix_is_still_available_and_is_exactly_the_old_behaviour():
    """Kept so a stored bounded reproduction can be rebuilt — the gate is on claiming from it,
    not on running it."""
    assert R.select_series(ALL, 10, selection="prefix") == ALL[:10]


def test_random_is_seeded_reproducible_and_is_not_the_prefix():
    a = R.select_series(ALL, 10, selection="random", seed=1)
    assert a == R.select_series(ALL, 10, selection="random", seed=1)
    assert a != R.select_series(ALL, 10, selection="random", seed=2)
    assert a != ALL[:10]
    assert len(a) == len(set(a)) == 10 and set(a) <= set(ALL)


def test_random_subsets_are_not_nested_across_sizes():
    """The nesting (5 ⊂ 10 ⊂ 20) is what let the loop 'confirm' a probe on the probe's own series
    (ST-4). A sample at each size is not nested, so that particular door closes as a side effect."""
    five = set(R.select_series(ALL, 5, selection="random", seed=1))
    ten = set(R.select_series(ALL, 10, selection="random", seed=1))
    assert not five <= ten


def test_asking_for_everything_returns_everything_whatever_the_rule():
    for sel in R.SERIES_SELECTIONS:
        assert R.select_series(ALL, 56, selection=sel, seed=1) == ALL
        assert R.select_series(ALL, None, selection=sel, seed=1) == ALL
        assert R.select_series(ALL, 99, selection=sel, seed=1) == ALL


def test_an_unknown_rule_is_refused_rather_than_defaulted():
    with pytest.raises(ValueError, match="unknown series_selection"):
        R.select_series(ALL, 5, selection="first_ten_please")


def test_stratified_spreads_the_sample_across_working_set_size_strata():
    sizes = {s: s * 100 for s in ALL}                  # size increases with id, so strata are ids
    pick = R.select_series(ALL, 8, selection="stratified", seed=1, sizes=sizes, n_strata=4)
    assert len(pick) == 8
    quartiles = [set(ALL[i * 14:(i + 1) * 14]) for i in range(4)]
    hits = [len([p for p in pick if p in q]) for q in quartiles]
    assert all(h >= 1 for h in hits), hits            # every stratum represented
    assert pick == R.select_series(ALL, 8, selection="stratified", seed=1, sizes=sizes)


def test_stratified_without_sizes_refuses_instead_of_falling_back():
    with pytest.raises(ValueError, match="per-series working-set sizes"):
        R.select_series(ALL, 5, selection="stratified", seed=1)


def test_stratification_on_a_model_outcome_is_not_offered():
    """The audit row suggested stratifying on baseline R². That is an outcome measured on the same
    held-out sets the run then evaluates on, which is ST-10's defect — so the rule does not exist
    and asking for it fails loudly rather than silently selecting on y."""
    assert "baseline_r2" not in R.SERIES_SELECTIONS
    with pytest.raises(ValueError):
        R.select_series(ALL, 5, selection="baseline_r2", seed=1)


# ------------------------------------------------------------------ the spec records the rule
def test_resolve_series_honours_an_explicit_list_untouched():
    splits = {"iterations": {"iteration_1": {"series": ALL}}}
    assert R.resolve_series(_spec(series=[9, 4, 1], iterations=[1]), splits) == [9, 4, 1]


def test_resolve_series_defaults_to_a_sample_not_a_prefix():
    splits = {"iterations": {"iteration_1": {"series": ALL}}}
    got = R.resolve_series(_spec(n_series=10, iterations=[1], seed=1), splits)
    assert got != ALL[:10] and len(got) == 10


def test_the_sample_seed_is_separable_from_the_run_seed():
    splits = {"iterations": {"iteration_1": {"series": ALL}}}
    a = R.resolve_series(_spec(n_series=10, iterations=[1], seed=1, series_sample_seed=42), splits)
    b = R.resolve_series(_spec(n_series=10, iterations=[1], seed=2, series_sample_seed=42), splits)
    assert a == b                                      # the series sample does not move with seed


def test_validation_rejects_an_unknown_rule_and_warns_on_a_prefix_subset():
    bad = validate_runspec(_spec(series_selection="nonsense"), unlocked_tier="tier_1")
    assert any("unknown series_selection" in e for e in bad.errors)
    warn = validate_runspec(_spec(n_series=10, series_selection="prefix"), unlocked_tier="tier_1")
    assert any("cannot carry a claim" in w for w in warn.warnings)
    assert not any("series_selection" in e for e in warn.errors)   # a probe still runs


# ------------------------------------------------------------------ the claim gate
def test_a_prefix_subset_accept_records_the_run_but_withholds_the_claim(tmp_path):
    cmp = {"candidate_model": "cnn", "baseline_model": "rf", "mean_delta": 0.031,
           "n_series": 10, "n_series_available": 56, "series_selection": "prefix"}
    entry = claim_registry.record(run_id="r1", proposal_id="p", status="accepted",
                                  comparison=cmp, claim_allowed="cnn beats rf",
                                  claims_dir=tmp_path)
    assert entry["claim"] is None
    assert "not a sample" in entry["claim_withheld"]
    assert entry["status"] == "accepted"               # the trail stays complete
    assert json.loads((tmp_path / "r1.json").read_text())["claim"] is None


def test_a_sampled_subset_accept_keeps_its_claim(tmp_path):
    cmp = {"n_series": 10, "n_series_available": 56, "series_selection": "random"}
    entry = claim_registry.record(run_id="r2", proposal_id="p", status="accepted",
                                  comparison=cmp, claim_allowed="cnn beats rf",
                                  claims_dir=tmp_path)
    assert entry["claim"] == "cnn beats rf" and entry["claim_withheld"] is None


def test_the_full_series_set_is_fine_however_it_was_ordered(tmp_path):
    """56 of 56 taken 'as a prefix' is the whole population, not a subset — the gate must not fire
    on the all-series runs that the audit's own fix recommends."""
    cmp = {"n_series": 56, "n_series_available": 56, "series_selection": "prefix"}
    entry = claim_registry.record(run_id="r3", proposal_id="p", status="accepted",
                                  comparison=cmp, claim_allowed="c", claims_dir=tmp_path)
    assert entry["claim"] == "c" and entry["claim_withheld"] is None


def test_older_records_without_a_selection_field_are_not_retroactively_invalidated(tmp_path):
    """Pre-2026-11-02 records have no `series_selection`; they were all prefix runs, but silently
    rewriting history here is not the mechanism — `docs/RETRACTIONS.md` is."""
    entry = claim_registry.record(run_id="r4", proposal_id="p", status="accepted",
                                  comparison={"n_series": 10}, claim_allowed="c",
                                  claims_dir=tmp_path)
    assert entry["claim"] == "c" and entry["claim_withheld"] is None
    assert entry["series_selection"] is None


def test_a_rejected_run_has_nothing_to_withhold(tmp_path):
    cmp = {"n_series": 10, "n_series_available": 56, "series_selection": "prefix"}
    entry = claim_registry.record(run_id="r5", proposal_id="p", status="rejected",
                                  comparison=cmp, claim_allowed=None, claims_dir=tmp_path)
    assert entry["claim"] is None and entry["claim_withheld"] is None


# ------------------------------------------------------------------ the third copy of the defect
def test_the_tournament_carries_the_selection_rule_to_the_registry(tmp_path):
    """`tournament` had its own `sorted(...)[:n_series]` and writes accepted claims directly, so
    the gate is only real if the rule reaches `claim_registry` from there too."""
    from seq2yield.experiments.tournament import Contender, TournamentResult, record_tournament
    board = [Contender(rank=1, model="cnn", r2=0.5, hyperparameters_source="default"),
             Contender(rank=2, model="rf", r2=0.45, hyperparameters_source="default",
                       delta_vs_winner=0.05, q_value=0.01, survives_fdr=True)]
    res = TournamentResult(dataset="ecoli", subregion=None, scope="per_series",
                           bootstrap_unit="series", train_size=1000, feature_set="one_hot",
                           family=["cnn", "rf"], winner="cnn", winner_significant=True,
                           runner_up="rf", min_delta=0.02, alpha=0.05, n_units=8,
                           leaderboard=board, series_selection="prefix", n_series_available=56)
    entry = record_tournament(res, claims_dir=tmp_path)
    assert entry["series_selection"] == "prefix" and entry["claim"] is None
    assert "not a sample" in entry["claim_withheld"]


def test_a_sampled_tournament_still_claims(tmp_path):
    from seq2yield.experiments.tournament import Contender, TournamentResult, record_tournament
    board = [Contender(rank=1, model="cnn", r2=0.5, hyperparameters_source="default"),
             Contender(rank=2, model="rf", r2=0.45, hyperparameters_source="default",
                       delta_vs_winner=0.05, q_value=0.01, survives_fdr=True)]
    res = TournamentResult(dataset="ecoli", subregion=None, scope="per_series",
                           bootstrap_unit="series", train_size=1000, feature_set="one_hot",
                           family=["cnn", "rf"], winner="cnn", winner_significant=True,
                           runner_up="rf", min_delta=0.02, alpha=0.05, n_units=8,
                           leaderboard=board, series_selection="random", n_series_available=56)
    entry = record_tournament(res, claims_dir=tmp_path)
    assert entry["claim"] is not None and entry["claim_withheld"] is None
