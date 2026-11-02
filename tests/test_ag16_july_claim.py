"""AG-16: the 2026-07-16 cnn-vs-mlp claim, re-measured and confirmed (DECISIONS #75, R-5 ledger).

The July record was `accepted` on a sorted 10-series prefix with `p_value: 0.0`, one model-init
seed, and a baseline from a registry the guard now refuses. Every one of those is a thing an audit
has condemned, and the obvious move was to retract it. Measuring first showed the opposite: the
FINDING holds at every train size on all 56 series. These tests pin both halves — that the result
still follows from the registry, and that the prefix form would be claim-withheld today — because
"we checked and it was fine" is worth nothing unless the check reruns.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

JULY = "2026-07-16-council-cnn-vs-mlp-203850"
RUN_ID = "2026-11-02-cnn-vs-mlp-seedmatched"
REGISTRY = "2026-11-02-full56-seeds"
RUN = ROOT / "experiments" / "runs" / RUN_ID
SIZES = [250, 500, 1000, 2000]
PRIMARY = 2000

pytestmark = pytest.mark.skipif(
    not (ROOT / "experiments" / "runs" / REGISTRY / "metrics.csv").exists(),
    reason="registry artifacts not present in this checkout")


def _card() -> dict:
    return json.loads((RUN / "run_card.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def registry():
    return pd.read_csv(ROOT / "experiments" / "runs" / REGISTRY / "metrics.csv")


@pytest.fixture(scope="module")
def recomputed(registry):
    from seq2yield.experiments import compare
    from seq2yield.experiments.run_spec import AcceptancePolicy
    policy = AcceptancePolicy(track="performance", baseline_run_id=REGISTRY,
                              baseline_model="mlp", comparison_train_size=PRIMARY)
    out = compare.compare_per_size(registry, registry, SIZES, "mlp", "cnn", policy, seed=0)
    return {r["train_size"]: r for r in out}


# --------------------------------------------------------------- the finding, recomputed not read
def test_cnn_beats_mlp_by_the_margin_at_every_train_size(recomputed):
    for size in SIZES:
        r = recomputed[size]
        assert r["status"] == "accepted", f"train {size}: {r['status']}/{r['margin_verdict']}"
        assert r["margin_verdict"] == "superior_by_margin"
        assert r["ci_one_sided"][0] >= r["margin"], (
            f"train {size}: lower limit {r['ci_one_sided'][0]:+.4f} no longer clears "
            f"the {r['margin']} margin")


def test_the_recorded_curve_still_follows_from_the_registry(recomputed):
    for size, stored in _card()["results"]["data_size_curve"].items():
        live = recomputed[int(size)]
        assert live["status"] == stored["status"], f"train {size}: status moved"
        assert live["margin_verdict"] == stored["margin_verdict"]
        assert live["mean_delta"] == pytest.approx(stored["mean_delta_r2"], abs=1e-6)


def test_the_july_candidate_reproduces_against_todays_registry(registry):
    """The load-bearing fact behind "confirmed, not retracted": the candidate is not approximately
    the same model, it is the same numbers. cnn is deterministic (unlike rf's n_jobs=-1), so this
    is exactly 0 or the claim of reproduction is wrong."""
    july_csv = ROOT / "experiments" / "runs" / JULY / "metrics.csv"
    if not july_csv.exists():
        pytest.skip("the July run's artifacts are not in this checkout")
    july = pd.read_csv(july_csv)
    cur = registry[(registry.model == "cnn") & (registry.model_seed == 1)]
    m = july.merge(cur, on=["iteration", "series", "train_size"], suffixes=("_jul", "_reg"))
    assert len(m) == len(july), f"only {len(m)} of {len(july)} July rows matched the registry"
    assert (m.r2_reg - m.r2_jul).abs().max() == 0.0, "the cnn candidate no longer reproduces"


def test_the_stale_registry_did_not_bias_this_particular_baseline(registry):
    """R-5 ARGUED that the July claim was safe because it scored against mlp, the family that never
    drifted. This measures it: the refused June registry and the current one give the same mlp
    value, so the ML-02 defect did not reach this comparison. It is the one thing that makes a
    claim measured against a refused control still trustworthy, and it is specific to mlp."""
    june = ROOT / "experiments" / "runs" / "2026-06-04-full56" / "metrics.csv"
    if not june.exists():
        pytest.skip("the June registry is not in this checkout")
    old = pd.read_csv(june)
    prefix = sorted(registry.series.unique())[:10]

    def mlp_mean(df):
        sub = df[(df.model == "mlp") & (df.train_size == PRIMARY) & (df.series.isin(prefix))]
        if "model_seed" in sub.columns:
            sub = sub[sub.model_seed == 1]
        return sub.groupby("series").r2.mean().mean()

    assert mlp_mean(old) == pytest.approx(mlp_mean(registry), abs=1e-9), (
        "mlp has drifted between the June and current registries, which would retroactively "
        "undermine R-5's defence of the June council claims")


# ------------------------------------------------------------------- SA-2, the part that transfers
def test_the_prefix_was_lucky_and_the_full_set_is_lower(recomputed, registry):
    """The defect that was real. The conclusion survives only because the effect is far above the
    margin; the inflation itself is what SA-2 is about and must stay visible."""
    from seq2yield.experiments.runner import per_series_r2_by_seed
    b = per_series_r2_by_seed(registry, PRIMARY, "mlp")
    c = per_series_r2_by_seed(registry, PRIMARY, "cnn")
    delta = (pd.concat(c.values(), axis=1).mean(axis=1)
             - pd.concat(b.values(), axis=1).mean(axis=1))
    prefix = sorted(registry.series.unique())[:10]
    assert delta.loc[prefix].mean() > delta.mean(), (
        "the first-10 prefix no longer overstates the all-56 delta, so the SA-2 wording in "
        "DECISIONS #75 and the loop-report banner needs re-measuring")
    assert recomputed[PRIMARY]["mean_delta"] == pytest.approx(delta.mean(), abs=1e-6)


def test_the_prefix_form_would_be_claim_withheld_today():
    """The gate is on the SELECTION RULE, not the subset size — a seeded random 10 has a sampling
    distribution and a sorted prefix does not. Asserted in both directions, because a gate that
    refuses every subset would be a different (and wrong) rule."""
    from seq2yield.experiments import claim_registry
    withheld = claim_registry._withheld(
        "accepted", {"series_selection": "prefix", "n_series": 10, "n_series_available": 56})
    assert withheld and "prefix" in withheld and "not a sample" in withheld

    assert claim_registry._withheld(
        "accepted", {"series_selection": "random", "n_series": 10,
                     "n_series_available": 56}) is None
    assert claim_registry._withheld(
        "accepted", {"series_selection": "all", "n_series": 56,
                     "n_series_available": 56}) is None


# ------------------------------------------------------------------------------- the record itself
def test_the_claim_is_recorded_with_everything_needed_to_quote_it():
    card = _card()
    claim = card["results"]["claim_allowed"]
    assert claim, "an accepted comparison that clears the margin must carry its claim text"
    # the train size, because the delta runs +0.1411 to +0.0690 across the curve
    assert "2000" in claim and "0.0690" in claim
    # and the scope limit that #74 established, in the claim itself rather than only the footnotes
    assert "RF" in claim and "margin" in claim

    ledger = ROOT / "experiments" / "claims" / "registry.jsonl"
    rows = [json.loads(ln) for ln in ledger.read_text(encoding="utf-8").splitlines() if ln.strip()]
    mine = [r for r in rows if r["run_id"] == RUN_ID]
    assert len(mine) == 1
    e = mine[0]
    assert e["status"] == "accepted" and e["claim"] and e["claim_withheld"] is None
    assert e["series_selection"] == "all" and e["n_series"] == 56
    assert e["margin_verdict"] == "superior_by_margin"
    for field in ("ci_method", "p_method", "split_strategy", "baseline_source", "provenance",
                  "ci_one_sided", "margin"):
        assert e.get(field) is not None, f"the record is missing {field}"
    assert isinstance(e["baseline_source"], str)


def test_the_july_record_is_left_identifiable_as_pre_contract():
    """Its null contract fields are what MARK it; rewriting an append-only ledger to tidy them
    would destroy the only signal that the row predates the contract."""
    ledger = ROOT / "experiments" / "claims" / "registry.jsonl"
    rows = [json.loads(ln) for ln in ledger.read_text(encoding="utf-8").splitlines() if ln.strip()]
    july = [r for r in rows if r["run_id"] == JULY]
    if not july:
        pytest.skip("the July ledger line is not in this checkout")
    e = july[0]
    assert e["status"] == "accepted" and e.get("claim") is None, (
        "the July row stated no claim text and must not acquire one retroactively")
    assert e.get("margin_verdict") is None and e.get("ci_method") is None, (
        "the July row has been back-filled, which destroys the signal that it is pre-contract")


def test_the_superseding_relationship_is_recorded_in_both_directions():
    """A re-measurement nobody can find from the original is not much use."""
    card = _card()
    assert card["results"]["supersedes_run_id"] == JULY
    stats = json.loads((RUN / "statistical_tests.json").read_text(encoding="utf-8"))
    assert stats["supersedes"]["run_id"] == JULY and stats["supersedes"]["why"]

    report = ROOT / "reports" / "static" / f"{JULY}_loop_report.md"
    if report.exists():
        head = report.read_text(encoding="utf-8")[:2200]
        assert RUN_ID in head, "the July report does not name its re-measurement"
        assert "CONFIRMED" in head and "98.8th percentile" in head
