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


# ------------------------------------------------ found by the post-close audit of #74 (2026-11-02)
def test_a_claim_record_cannot_blank_the_dashboard():
    """Found by the full suite after #74 was committed, and it was MY regression.

    `claim.baseline_source` is declared TEXT and had been null in all 22 preceding records, so the
    column was never exercised. The first record to populate it carried a dict, `sqlite3` refused to
    bind it, and that aborted `ingest_all` — taking every table and the whole dashboard with it, not
    just the one row. `_put` now coerces structured values, by the same rule as `audit_log`'s
    coercing `default=` (DECISIONS #69): the layer that records what happened must not be the layer
    that fails."""
    import sqlite3

    from orchestration import store
    con = sqlite3.connect(":memory:")
    for stmt in store.SCHEMA:
        con.execute(stmt)
    # every JSON shape a record can legitimately arrive as
    store._put(con, "claim", {
        "run_id": "t-dict", "ts": "now", "candidate_model": "cnn", "baseline_model": "rf",
        "baseline_source": {"kind": "registry", "run_id": "r"},   # the shape that broke it
        "split_strategy": ["a", "b"], "ci_excludes_zero": True, "margin": 0.02})
    row = con.execute("SELECT baseline_source, split_strategy, ci_excludes_zero FROM claim "
                      "WHERE run_id='t-dict'").fetchone()
    assert json.loads(row[0]) == {"kind": "registry", "run_id": "r"}, "the value must survive"
    assert json.loads(row[1]) == ["a", "b"]
    assert row[2] == 1, "a bool must store as an int, not raise"


def test_baseline_source_is_a_string_in_every_ledger_row():
    """The shape contract, which nothing enforced. `execution_harness` writes this field as a
    string in all three of its branches ("registry:<id>" / "in_run" / "in_run (registry refused:
    ...)"), and both `run_dashboard` and the loop report render it as text — so a structured value
    is not corruption but it IS the only non-uniform row in the ledger, displayed as a JSON blob
    where every other row shows a control id. #74's record carried a dict until this caught it."""
    ledger = ROOT / "experiments" / "claims" / "registry.jsonl"
    rows = [json.loads(ln) for ln in ledger.read_text(encoding="utf-8").splitlines() if ln.strip()]
    bad = [(r["run_id"], type(r["baseline_source"]).__name__) for r in rows
           if r.get("baseline_source") is not None and not isinstance(r["baseline_source"], str)]
    assert not bad, f"baseline_source must be a string: {bad}"


def test_the_run_cards_limitations_do_not_claim_a_grouping_ecoli_does_not_have():
    """The audit's own finding, as a test. The limitations block previously said "6 of 56 series
    carry the grouped structure the mixed-effects estimator handles (ICC 0.864)" — a count with no
    source, and a statistic belonging to #72's per-series DIFFICULTY decomposition. Checked against
    the DatasetSpec rather than remembered, which is the point."""
    from seq2yield.data.datasets import spec
    s = spec("ecoli")
    assert s.family_col is None and s.family_distance is None and not s.replicate_cols, (
        "E. coli now declares a grouping, so the limitations wording needs revisiting rather than "
        "this assertion being relaxed")
    text = " ".join(_card()["limitations"])
    claim, correction = text.split("CORRECTION", 1) if "CORRECTION" in text else (text, "")
    assert "0.864" not in claim, "the difficulty ICC is back in a limitation that is not about it"
    assert "6 of 56" not in claim, "the unsourced count is back"


def test_every_claim_column_the_store_maps_survives_a_round_trip():
    """The generalisation of the bug above, which is the part worth keeping.

    `baseline_source` was not specially unlucky: FIVE claim columns — baseline_source,
    split_strategy, ci_method, margin_verdict, margin — were null in all 22 records preceding #74's,
    because they were added to the contract by the audits of #64/#65/#66/#69 and no run had recorded
    one since. The entire provenance-and-margin block reached the schema and the dashboard without a
    single record exercising it, and the first one to try took the page down. This drives a full
    record through `ingest_claims` so a column cannot be dead AND unproven at the same time."""
    import sqlite3

    from orchestration import store
    con = sqlite3.connect(":memory:")
    con.row_factory = sqlite3.Row          # as `store.connect` sets it; dict(row) needs it
    for stmt in store.SCHEMA:
        con.execute(stmt)

    live = [json.loads(ln) for ln
            in (ROOT / "experiments" / "claims" / "registry.jsonl").read_text(
                encoding="utf-8").splitlines() if ln.strip()]
    record = next(r for r in live if r["run_id"] == RUN_ID)

    store._put(con, "claim", {
        "run_id": record["run_id"], "ts": record["ts"],
        "candidate_model": record["candidate_model"], "baseline_model": record["baseline_model"],
        "mean_delta_r2": record["mean_delta_r2"],
        "ci_excludes_zero": int(bool(record["ci_excludes_zero"])),
        "p_value": record["p_value"], "bootstrap_unit": record["bootstrap_unit"],
        "dataset": record["dataset"], "train_size": record["train_size"],
        "status": record["status"], "claim": record["claim"],
        "baseline_source": record["baseline_source"], "split_strategy": record["split_strategy"],
        "code_sha": (record.get("provenance") or {}).get("git_sha"),
        "ci_method": record["ci_method"], "margin_verdict": record["margin_verdict"],
        "margin": record["margin"]})

    row = con.execute("SELECT * FROM claim WHERE run_id=?", (RUN_ID,)).fetchone()
    got = dict(row)
    # the five that were dead before this run: each must be present AND non-null
    for col in ("baseline_source", "split_strategy", "ci_method", "margin_verdict", "margin"):
        assert got[col] is not None, f"{col} did not survive the round trip"
    assert got["margin"] == 0.02
    assert got["margin_verdict"] == "superior_but_below_margin"
    assert got["ci_method"] == "t_two_component"
    assert got["status"] == "rejected" and got["claim"] is None
    assert str(got["baseline_source"]).startswith("registry:")
