"""Family-aware (cluster) paired bootstrap — audit ST-7, the follow-up to SA-1/ML-03.

Holding whole design families out fixed WHICH sequences are scored. It did not fix how they are
COUNTED: a 4-member family is one observation drawn four times, so resampling its members as if
independent gives an interval that is too narrow and a p-value that is too small. These tests pin
(1) that the old sequence-level path is untouched, (2) that the grouped path is a strict
generalization of it, (3) that it actually widens the interval on clustered data, and (4) that it
cuts the false-positive rate the sequence unit produces on an exactly-null comparison.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seq2yield.statistics import bootstrap as B  # noqa: E402


def _simple(seed=0, n=300):
    rng = np.random.default_rng(seed)
    y = rng.uniform(0, 100, n)
    return y, y + rng.normal(0, 3, n), y + rng.normal(0, 25, n)


def _clustered_null(rng, n_fam=24, k=20, strong=0.4, weak=0.9):
    """A comparison that is null BY CONSTRUCTION yet clustered.

    Each family picks which model happens to fit it better; the two models are exchangeable across
    families, so the true ΔR² is 0 — but within a family the delta is large and same-signed, which
    is exactly the structure a designed library has and the sequence-level bootstrap cannot see."""
    fam = np.repeat(np.arange(n_fam), k)
    y = rng.normal(0, 1, n_fam * k)
    eps = rng.normal(0, 1, n_fam * k)
    cand_better = (rng.random(n_fam) < 0.5)[fam]
    cand = y + eps * np.where(cand_better, strong, weak)
    base = y + eps * np.where(cand_better, weak, strong)
    return y, cand, base, fam


# ----------------------------------------------------- the old path must not move at all
def test_groups_none_is_the_old_sequence_bootstrap_bit_for_bit():
    """Legacy run-cards have to stay reproducible, so the ungrouped path is pinned against a
    verbatim copy of the pre-change implementation — not merely 'close'."""
    y, good, bad = _simple()

    def old_paired(y_true, pc, pb, n_boot=2000, alpha=0.05, seed=0):
        y_true, pc, pb = (np.asarray(a, float) for a in (y_true, pc, pb))
        n = len(y_true)
        rng = np.random.default_rng(seed)
        d = [B._r2(y_true[i], pc[i]) - B._r2(y_true[i], pb[i])
             for i in (rng.integers(0, n, n) for _ in range(n_boot))]
        lo, hi = np.nanquantile(np.array(d), [alpha / 2, 1 - alpha / 2])
        return [float(lo), float(hi)]

    def old_single(y_true, y_pred, n_boot=2000, alpha=0.05, seed=0):
        y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
        n = len(y_true)
        rng = np.random.default_rng(seed)
        b = np.array([B._r2(y_true[i], y_pred[i])
                      for i in (rng.integers(0, n, n) for _ in range(n_boot))])
        lo, hi = np.nanquantile(b, [alpha / 2, 1 - alpha / 2])
        return [float(lo), float(hi)]

    new = B.paired_bootstrap_r2(y, good, bad, seed=0)
    assert new["ci"] == old_paired(y, good, bad)
    assert B.bootstrap_r2_ci(y, good, seed=0)["ci"] == old_single(y, good)
    assert new["unit"] == "sequence" and new["n_units"] == 300


def test_one_label_per_row_reduces_exactly_to_the_sequence_bootstrap():
    """The grouped path is a generalization, not a different estimator: with singleton families
    (a truly random library) drawing families IS drawing rows, down to the same draws."""
    y, good, bad = _simple()
    plain = B.paired_bootstrap_r2(y, good, bad, seed=0)
    grouped = B.paired_bootstrap_r2(y, good, bad, groups=np.arange(len(y)), seed=0)
    assert grouped["ci"] == plain["ci"] and grouped["p_value"] == plain["p_value"]
    assert grouped["unit"] == "family" and grouped["n_units"] == len(y)
    single = B.bootstrap_r2_ci(y, good, groups=np.arange(len(y)), seed=0)
    assert single["ci"] == B.bootstrap_r2_ci(y, good, seed=0)["ci"]


def test_the_point_estimate_never_depends_on_the_unit():
    """Only the uncertainty changes. If the mean delta moved, the fix would be rewriting results
    rather than re-describing their precision."""
    rng = np.random.default_rng(3)
    y, cand, base, fam = _clustered_null(rng)
    a = B.paired_bootstrap_r2(y, cand, base, n_boot=200, seed=0)
    b = B.paired_bootstrap_r2(y, cand, base, groups=fam, n_boot=200, seed=0)
    assert a["mean_delta"] == b["mean_delta"]
    assert a["n"] == b["n"] == len(y)


# ----------------------------------------------------- what the family unit actually buys
def test_a_designed_library_gets_a_wider_interval():
    rng = np.random.default_rng(7)
    y, cand, base, fam = _clustered_null(rng)
    seq = B.paired_bootstrap_r2(y, cand, base, n_boot=600, seed=0)
    grp = B.paired_bootstrap_r2(y, cand, base, groups=fam, n_boot=600, seed=0)
    width = lambda r: r["ci"][1] - r["ci"][0]                              # noqa: E731
    assert width(grp) > 2 * width(seq)                                     # ~sqrt(k) = sqrt(20)
    assert grp["n_units"] == 24 and seq["n_units"] == len(y)


def test_the_sequence_unit_declares_a_null_difference_significant_far_too_often():
    """The cost of the wrong unit, as a false-positive rate. Both models are exchangeable, so
    every 'significant' verdict here is a type-I error; the nominal rate is 5%."""
    seq_hits = grp_hits = 0
    trials = 30
    for t in range(trials):
        rng = np.random.default_rng(1000 + t)
        y, cand, base, fam = _clustered_null(rng)
        seq_hits += B.paired_bootstrap_r2(y, cand, base, n_boot=200, seed=t)["excludes_zero"]
        grp_hits += B.paired_bootstrap_r2(y, cand, base, groups=fam, n_boot=200,
                                          seed=t)["excludes_zero"]
    assert seq_hits / trials > 0.5                      # badly over-confident on clustered data
    assert grp_hits / trials <= 0.2                     # family unit stays near nominal
    assert grp_hits < seq_hits / 2


def test_identical_models_are_not_significant_under_either_unit():
    """Sanity in the other direction: the grouped bootstrap is not merely 'never significant'."""
    rng = np.random.default_rng(5)
    y, cand, _, fam = _clustered_null(rng)
    tie = B.paired_bootstrap_r2(y, cand, cand, groups=fam, n_boot=200, seed=0)
    assert tie["mean_delta"] == 0.0 and not tie["excludes_zero"]
    strong = B.paired_bootstrap_r2(y, cand, y + rng.normal(0, 4, len(y)), groups=fam,
                                   n_boot=200, seed=0)
    assert strong["mean_delta"] > 0 and strong["excludes_zero"]   # a real gap still lands


# ----------------------------------------------------- degenerate cases fail loud, not silent
def test_a_single_family_refuses_to_report_a_ci():
    """With one cluster every resample is the identical sample: zero variance, and any non-zero
    delta would come out 'significant'. It must refuse instead."""
    y, good, bad = _simple()
    out = B.paired_bootstrap_r2(y, good, bad, groups=np.zeros(len(y)), seed=0)
    assert out["n_units"] == 1 and np.isnan(out["ci"]).all()
    assert out["excludes_zero"] is False and out["p_value"] == 1.0 and "note" in out
    assert out["mean_delta"] > 0                       # the estimate is still reported, just unsafe
    one = B.bootstrap_r2_ci(y, good, groups=np.zeros(len(y)), seed=0)
    assert np.isnan(one["ci"]).all() and "note" in one


def test_a_single_series_refuses_too():
    """Same failure mode in the per-series bootstrap, which had the same hole."""
    out = B.paired_bootstrap_ci([0.5], [0.9], seed=0)
    assert out["n_units"] == 1 and out["excludes_zero"] is False and "note" in out
    ok = B.paired_bootstrap_ci([0.5] * 8, [0.62] * 8, seed=0)
    assert ok["unit"] == "series" and ok["n_units"] == 8 and ok["excludes_zero"] is True


def test_malformed_group_labels_are_refused():
    y, good, bad = _simple(n=10)
    with pytest.raises(ValueError):                          # wrong length
        B.paired_bootstrap_r2(y, good, bad, groups=np.arange(9))
    with pytest.raises(ValueError):                          # unlabelled row: caller must decide
        B.paired_bootstrap_r2(y, good, bad, groups=[1, 2, 3, np.nan, 5, 6, 7, 8, 9, 10])
    with pytest.raises(ValueError):
        B.paired_bootstrap_r2(y, good, bad, groups=[1, 2, 3, None, 5, 6, 7, 8, 9, 10])


def test_string_family_labels_work():
    """Families come from a data column as often as from Hamming components."""
    y, good, bad = _simple(n=60)
    labels = [f"series_{i // 10}" for i in range(60)]
    out = B.paired_bootstrap_r2(y, good, bad, groups=labels, n_boot=200, seed=0)
    assert out["n_units"] == 6 and out["unit"] == "family"


# ----------------------------------------------------- the unit reaches the run-card
def test_the_pooled_harness_records_the_family_unit(monkeypatch, tmp_path):
    """The harness must resample by family AND say so, since a family-unit CI is not comparable
    with a sequence-unit one (the C3 fence)."""
    from orchestration import execution_harness as EH
    from seq2yield.experiments import pooled_runner as PR

    rng = np.random.default_rng(0)
    y, cand_pred, base_pred, fam = _clustered_null(rng, n_fam=12, k=10)

    def fake_run_pooled(spec, model_family=None, **kw):
        mf = model_family or spec.model_family
        pred = cand_pred if mf == "cnn" else base_pred
        return {"y_test": y, "test_series": None, "test_families": fam,
                "preds": {500: pred}, "n_train_full": 900}

    monkeypatch.setattr(PR, "run_pooled", fake_run_pooled)
    monkeypatch.setattr(PR, "split_report", lambda spec, **k: {"split_strategy": "grouped_holdout"})
    monkeypatch.setattr(EH, "_attach_diagnostics", lambda *a, **k: None)

    from seq2yield.experiments.run_spec import AcceptancePolicy, RunSpec
    spec = RunSpec(run_id="t", dataset="yeast", model_family="cnn", train_sizes=[500],
                   acceptance_policy=AcceptancePolicy(track="performance", baseline_model="rf"))
    out = EH._run_pooled(spec, tmp_path, type("V", (), {"warnings": []})())
    cmp = out["comparison"]
    assert cmp["bootstrap_unit"] == "family" and cmp["n_bootstrap_units"] == 12
    assert cmp["n_test"] == 120                        # rows, not units — both are recorded


def test_the_tournament_resamples_families_when_the_dataset_has_them(monkeypatch):
    from seq2yield.data.cleaning import SEQ_COL, TARGET_COL
    from seq2yield.experiments import pooled_runner as PR
    from seq2yield.experiments import tournament as T

    rng = np.random.default_rng(2)
    y, cand, base, fam = _clustered_null(rng, n_fam=10, k=8)
    frame = pd.DataFrame({SEQ_COL: ["A" * 50] * len(y), TARGET_COL: y, PR.FAMILY_COL: fam})
    monkeypatch.setattr(PR, "holdout", lambda spec, **k: (frame, frame))
    monkeypatch.setattr(T, "_seq_unit_family", lambda *a, **k: (
        {"cnn": {"pred": cand, "source": "default", "hparams": {}},
         "rf": {"pred": base, "source": "default", "hparams": {}}}, y))
    res = T.run_tournament("yeast", family=["cnn", "rf"], n_boot=200)
    assert res.bootstrap_unit == "family" and res.n_units == 10      # units, not the 80 rows


def test_a_subregion_tournament_keeps_the_grouping(monkeypatch):
    """`scope_frame` attaches families BEFORE a C6 strata filter, so a subregion run does not
    silently fall back to the stratified split and the sequence unit."""
    from seq2yield.data import strata
    from seq2yield.data.cleaning import SEQ_COL, SERIES_COL, TARGET_COL
    from seq2yield.experiments import pooled_runner as PR

    rng = np.random.default_rng(4)
    n = 400
    frame = pd.DataFrame({
        SEQ_COL: ["".join(rng.choice(list("ACGT"), 80)) for _ in range(n)],
        TARGET_COL: rng.normal(0, 1, n),
        SERIES_COL: np.repeat(np.arange(20), n // 20)})
    monkeypatch.setattr(PR, "_frame", lambda d: frame)
    scoped = PR.scope_frame("yeast")
    assert PR.FAMILY_COL in scoped.columns                  # yeast declares family_col=mut_series
    sub = strata.filter(scoped, "yeast", "gc_bin=high")
    assert len(sub) and PR.FAMILY_COL in sub.columns        # the label rides along on the rows
    assert PR.families_of(sub) is not None
    train, test = PR.holdout_frame(sub, seed=0)
    assert not set(train[PR.FAMILY_COL]) & set(test[PR.FAMILY_COL])


# ----------------------------------------------------- the count is observable on the run-card
def test_the_unit_count_is_a_diagnostic_signal_and_trips_a_flag():
    from seq2yield.diagnostics import critic, signals as S
    clustered = S.resampling_units(np.repeat(np.arange(5), 80), unit="series")
    assert clustered["n_units"] == 5 and clustered["n_rows"] == 400
    assert clustered["rows_per_unit"] == 80.0 and clustered["clustering"] > 0.98
    flags = critic.evaluate({"bootstrap_units": clustered})
    flag = next(f for f in flags if f["id"] == "too_few_bootstrap_units")
    assert flag["severity"] == "medium" and flag["blocking"] is False      # advisory
    random_lib = S.resampling_units(np.arange(400), unit="sequence")
    assert random_lib["clustering"] == 0.0
    assert "too_few_bootstrap_units" not in [
        f["id"] for f in critic.evaluate({"bootstrap_units": random_lib})]


def test_real_yeast_comparison_is_resampled_by_family(require_data):
    """On the real frame: the held-out rows carry families, the unit count is the family count
    (not the row count), and the interval is wider than the sequence-level one."""
    require_data("yeast")
    from types import SimpleNamespace

    from seq2yield.experiments import pooled_runner as PR
    spec = SimpleNamespace(dataset="yeast", seed=1, feature_set="one_hot",
                           feature_scaling="none", train_sizes=[500], sampling_policy="random",
                           hyperparameters=None, model_family="rf")
    out = PR.run_pooled(spec)
    fams = out["test_families"]
    assert fams is not None and len(set(fams)) < len(fams)      # really clustered
    y, pred = out["y_test"], out["preds"][500]
    worse = pred * 0.5 + float(np.mean(y)) * 0.5                # a deliberately degraded twin
    seq = B.paired_bootstrap_r2(y, pred, worse, n_boot=300, seed=1)
    grp = B.paired_bootstrap_r2(y, pred, worse, groups=fams, n_boot=300, seed=1)
    assert grp["n_units"] == len(set(fams)) and grp["n"] == len(y)
    assert (grp["ci"][1] - grp["ci"][0]) > (seq["ci"][1] - seq["ci"][0])
