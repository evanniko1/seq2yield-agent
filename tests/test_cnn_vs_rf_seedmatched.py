"""The E. coli cnn-vs-rf re-decision on the seed-matched registry (DECISIONS #74, R-5).

This is a RESULT guard, not a file guard. A run-card that says `rejected / superior_but_below_margin`
is worth nothing if the verdict no longer follows from the registry it was computed on, so the
comparison is recomputed here from `metrics.csv` through the project's own estimator and acceptance
rule and checked against what was recorded. If the estimator changes, the registry changes, or the
margin moves, this fails and the record has to be re-decided rather than quietly inherited — which
is the failure R-5 is about (a claim outliving the baseline that produced it).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RUN_ID = "2026-11-02-cnn-vs-rf-seedmatched"
RUN = ROOT / "experiments" / "runs" / RUN_ID
REGISTRY = "2026-11-02-full56-seeds"
SIZES = [250, 500, 1000, 2000]
PRIMARY = 2000

pytestmark = pytest.mark.skipif(
    not (ROOT / "experiments" / "runs" / REGISTRY / "metrics.csv").exists(),
    reason="registry artifacts not present in this checkout")


def _card() -> dict:
    return json.loads((RUN / "run_card.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def recomputed():
    from seq2yield.experiments import compare
    from seq2yield.experiments.run_spec import AcceptancePolicy
    df = pd.read_csv(ROOT / "experiments" / "runs" / REGISTRY / "metrics.csv")
    policy = AcceptancePolicy(track="performance", baseline_run_id=REGISTRY,
                              baseline_model="rf", comparison_train_size=PRIMARY)
    out = compare.compare_per_size(df, df, SIZES, "rf", "cnn", policy, seed=0)
    return {r["train_size"]: r for r in out}


def test_the_recorded_verdict_still_follows_from_the_registry(recomputed):
    """The whole point: recompute, do not trust the stored number."""
    card = _card()
    assert card["status"] == "rejected"
    for size, stored in card["results"]["data_size_curve"].items():
        live = recomputed[int(size)]
        assert live["status"] == stored["status"], f"train {size}: status moved"
        assert live["margin_verdict"] == stored["margin_verdict"], f"train {size}: verdict moved"
        assert live["mean_delta"] == pytest.approx(stored["mean_delta_r2"], abs=1e-6)
        for a, b in zip(live["ci_one_sided"], stored["ci_one_sided"]):
            assert a == pytest.approx(b, abs=1e-6)


def test_the_primary_comparison_is_a_conclusive_negative_not_an_unresolved_one(recomputed):
    """`superior_but_below_margin` is a specific claim: better than the baseline AND the gap shown
    to sit inside the margin. Both halves are asserted, because a drift to
    `superior_margin_unresolved` would weaken the result while keeping the same `rejected` status."""
    r = recomputed[PRIMARY]
    assert r["status"] == "rejected"
    assert r["margin_verdict"] == "superior_but_below_margin"
    assert r["mean_delta"] > 0 and r["ci_excludes_zero"]
    lo, hi = r["ci_one_sided"]
    assert lo > 0, "the lower limit must still exclude zero"
    assert hi < r["margin"], "the gap must still be ESTABLISHED to lie inside the margin"


def test_the_direction_holds_at_every_size_and_every_seed(recomputed):
    """R-5's surviving half. Twelve (size x seed) deltas, all positive, signs agreeing."""
    for size in SIZES:
        r = recomputed[size]
        assert r["mean_delta"] > 0, f"train {size}: cnn no longer ahead"
        assert r["ci_excludes_zero"], f"train {size}: zero no longer excluded"
        assert r["seed_signs_agree"], f"train {size}: model-init seeds disagree in sign"
        assert len(r["seed_deltas"]) == 3
        assert all(d > 0 for d in r["seed_deltas"]), f"train {size}: a seed went negative"


def test_the_margin_is_established_at_no_train_size(recomputed):
    """The load-bearing negative. An `accepted` here would mean the one-sided lower limit cleared
    0.02, which is the claim R-5 withdrew; it must not come back by accident."""
    for size in SIZES:
        r = recomputed[size]
        assert r["status"] != "accepted", f"train {size}: the withdrawn claim is back"
        assert r["margin_verdict"] != "superior_by_margin"
        assert r["ci_one_sided"][0] < r["margin"]


def test_both_arms_carry_three_seeds_so_the_interval_has_both_components(recomputed):
    """Without this the comparison silently reverts to a series-only interval (audit ML-01b), which
    is what every pre-2026-11-02 figure in R-5 was."""
    for size in SIZES:
        r = recomputed[size]
        # NB `n_seeds` is an int on the two-component path and a {candidate, baseline} dict on
        # `compare_per_size`'s single-seed fallback. An int here is itself the evidence that the
        # multiseed estimator ran; the dict would mean it did not (logged as AG-17).
        assert r["n_seeds"] == 3, f"train {size}: {r.get('n_seeds')!r}"
        assert r["seed_design"] == "paired"
        assert r["se_seed"] > 0 and r["se_unit"] > 0
        assert r["se_total"] >= max(r["se_unit"], r["se_seed"])
        assert r["ci_method"] == "t_two_component"


def test_series_heterogeneity_dominates_initialisation_at_every_size(recomputed):
    """DECISIONS #74's second finding, and the reason more seeds is not where precision would come
    from. Asserted as the inequality, not the four measured values, so it tests the claim rather
    than pinning noise."""
    for size in SIZES:
        r = recomputed[size]
        assert r["se_seed"] < r["se_unit"], (
            f"train {size}: se_seed {r['se_seed']:.5f} >= se_unit {r['se_unit']:.5f} — the "
            "'seeds are not where the imprecision lives' conclusion no longer holds")


def test_the_cnn_advantage_decays_with_training_data(recomputed):
    """The other new finding. Quoting `cnn > rf` without its train size is the misleading version
    of a true statement, so the decay is pinned: the largest size must show the smallest gap."""
    deltas = {s: recomputed[s]["mean_delta"] for s in SIZES}
    assert deltas[2000] == min(deltas.values()), deltas
    assert deltas[2000] < deltas[500], deltas


def test_no_claim_is_recorded_for_this_run():
    """A rejected comparison may not leave a claim behind, and the REASON must travel with it
    rather than being inferable from a missing field."""
    card = _card()
    res = card["results"]
    assert res["claim_allowed"] is None
    assert res["claim_withheld_reason"] and "margin" in res["claim_withheld_reason"]

    ledger = ROOT / "experiments" / "claims" / "registry.jsonl"
    rows = [json.loads(ln) for ln in ledger.read_text(encoding="utf-8").splitlines() if ln.strip()]
    mine = [r for r in rows if r["run_id"] == RUN_ID]
    assert len(mine) == 1, f"expected exactly one ledger row for this run, got {len(mine)}"
    e = mine[0]
    # the ledger's field is `claim` (set only for an accepted run with nothing withheld);
    # `claim_withheld` carries SA-2's prefix refusal, which does not apply to an all-56 comparison
    assert e["status"] == "rejected" and e["claim"] is None
    assert e["claim_withheld"] is None
    assert e["margin_verdict"] == "superior_but_below_margin" and e["margin"] == 0.02
    # the fields whose absence identifies a pre-contract record must all be PRESENT here
    for field in ("ci_method", "p_method", "split_strategy", "series_selection", "n_series",
                  "baseline_source", "provenance", "ci_one_sided"):
        assert e.get(field) is not None, f"the record is missing {field}"
    assert e["series_selection"] == "all" and e["n_series"] == 56


def test_the_ledger_holds_no_accepted_ecoli_cnn_over_rf_claim():
    """R-5 withdrew it and #74 re-decided it seed-matched. It must not reappear — including from a
    prefix subset, which is how the inflated version was measured in the first place."""
    ledger = ROOT / "experiments" / "claims" / "registry.jsonl"
    rows = [json.loads(ln) for ln in ledger.read_text(encoding="utf-8").splitlines() if ln.strip()]
    offenders = [
        (r["run_id"], r.get("n_series"), r.get("mean_delta_r2")) for r in rows
        if r.get("status") == "accepted" and r.get("dataset") == "ecoli"
        and r.get("candidate_model") == "cnn" and r.get("baseline_model") == "rf"
        and r.get("claim")]
    assert not offenders, f"the withdrawn cnn>rf claim is live in the ledger: {offenders}"


def test_the_run_card_names_its_baseline_and_that_baseline_is_the_live_registry():
    """A comparison whose control cannot be identified is the ML-02 failure. The baseline here is
    the canonical registry, so the pointer and the run-card must agree."""
    from orchestration import baseline_guard
    card = _card()
    assert card["results"]["baseline_run_id"] == REGISTRY == baseline_guard.registry_run_id()
    stamp = (card.get("environment") or {}).get("numerics") or {}
    assert stamp.get("fingerprint"), "the run-card carries no numerics fingerprint"
