"""The insight/dissection rebuild: the "why" never conditions on the observed target except
through an out-of-fold prediction (audit SA-6 / ST-8 / ML-10).

Each test pins one claim made about the rebuild, and is written to FAIL if the corresponding defect
comes back. Where a test pins a defect, it also computes what the defective rule WOULD have said on
the same data, so the contrast is asserted rather than asserted-about.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seq2yield.insight import dissect, phase  # noqa: E402
from seq2yield.statistics import selection  # noqa: E402

ALPHA = 0.05


# ------------------------------------------------------------------- helpers ---
def _frame(rows):
    """rows: list of (series, model, train_size, r2, rmse[, n])."""
    cols = ["series", "model", "train_size", "r2", "rmse", "n"]
    rows = [r if len(r) == 6 else (*r, 100) for r in rows]
    df = pd.DataFrame(rows, columns=cols)
    df["iteration"] = "iteration_1"
    return df


def _two_group_pooled(seed=0, n=60):
    """A pooled test set whose two neighbourhoods invert R² against RMSE.

    Group A (GC-rich): tiny Var(y), errors LARGER than that variance -> R² far below 0, small RMSE.
    Group B (AT-rich): large Var(y), much larger absolute errors -> R² near 0.9, large RMSE.
    Any ranking that divides by the unit's own Var(y) calls A the hardest; A has 1/6 of B's error.
    """
    rng = np.random.default_rng(seed)
    a_seq = ["GCGCGCGCGCGCGCGCGCGC" * 3] * n
    b_seq = ["ATATATATATATATATATAT" * 3] * n
    ya = rng.normal(0, 0.1, n)
    yb = rng.normal(0, 3.0, n)
    pa = ya + rng.normal(0, 0.15, n)
    pb = yb + rng.normal(0, 1.0, n)
    train = (["GCGCGCGCGCGCGCGCGCGC" * 3] * 100) + (["ATATATATATATATATATAT" * 3] * 100)
    return (train, a_seq + b_seq, np.concatenate([ya, yb]), np.concatenate([pa, pb]), n)


# =============================================== SA-6: no target moments, named scale ===
def test_target_moment_covariate_is_dropped_by_name_even_when_it_wins():
    """SA-6: `expr_std` is R²'s denominator. Here it is the covariate most correlated with
    difficulty, so an un-guarded max-|r| selector MUST pick it; the rebuilt one drops it."""
    n = 30
    rng = np.random.default_rng(1)
    diff = pd.Series(rng.normal(size=n))
    cov = pd.DataFrame({
        "expr_std": diff.to_numpy() + rng.normal(0, 0.1, n),       # near-perfect, target-derived
        "expr_mean": diff.to_numpy() + rng.normal(0, 0.3, n),      # strong, target-derived
        "gc": rng.normal(size=n),                                  # pure noise
    })
    # the defective rule, on the same data: max |r| over the whole family
    rs = {c: abs(float(np.corrcoef(cov[c], diff)[0, 1])) for c in cov.columns}
    assert max(rs, key=rs.get) == "expr_std", "setup: expr_std must win an unguarded selection"

    qs = dissect.correlate_difficulty(diff, cov, "t", scale=dissect.SCALE_RMSE,
                                      higher_is_harder=True)
    assert len(qs) == 1
    ev = qs[0].evidence
    assert ev["covariate"] != "expr_std" and ev["covariate"] != "expr_mean"
    assert "expr_std" in ev["dropped_covariates"] and "expr_mean" in ev["dropped_covariates"]
    assert "SA-6" in ev["dropped_covariates"]["expr_std"]
    assert "expr_std" not in ev["candidate_family"]
    assert ev["reported"] is False            # only noise is left, so nothing is reported


def test_covariate_builders_emit_no_target_derived_column():
    """SA-6 at the source: the per-unit covariate tables are sequence/design only. The pooled
    builder still ACCEPTS y_true (callers pass it) and must ignore it."""
    train, seqs, y, _pred, n = _two_group_pooled()
    labels = np.array([0] * n + [1] * n)
    cov = dissect._neighborhood_covariates(seqs, labels, y, train_sequences=train)
    assert not [c for c in cov.columns if dissect._is_target_derived(c)]
    assert {"gc", "gc_sd", "length_cv", "seq_entropy", "kmer_dispersion", "n_rows",
            "train_kmer_dist"} == set(cov.columns)
    # and the table does not change when the targets are replaced wholesale
    other = dissect._neighborhood_covariates(seqs, labels, y * 100 + 7, train_sequences=train)
    pd.testing.assert_frame_equal(cov, other)


def test_difficulty_uses_an_error_scale_and_every_question_names_it():
    """SA-6: with an error column present, difficulty is RMSE and the scale is on the question.
    Series 1 has the worst R² and the SMALLEST error; series 3 has the largest error."""
    df = _frame([(1, "rf", 500, -2.0, 0.20), (1, "mlp", 500, -2.1, 0.21),
                 (2, "rf", 500, 0.50, 0.60), (2, "mlp", 500, 0.48, 0.62),
                 (3, "rf", 500, 0.80, 1.40), (3, "mlp", 500, 0.79, 1.42),
                 (4, "rf", 500, 0.60, 0.55), (4, "mlp", 500, 0.61, 0.54),
                 (5, "rf", 500, 0.55, 0.50), (5, "mlp", 500, 0.54, 0.52)])
    qs = dissect.dissect_metrics(df, "t")
    assert qs and all(q.difficulty_scale == dissect.SCALE_RMSE for q in qs)
    hard = next(q for q in qs if q.kind == "series_difficulty")
    # `ranked` is the difficulty ORDER, which is what this test is about: series 3 has the largest
    # error and series 1 the worst R2 with the smallest error, so an error scale must put 3 on the
    # list and leave 1 off it. `series` is the subset the permutation null ESTABLISHES, and with
    # 5 units in 2 blocks it cannot establish anything - which is itself correct (#71's residual,
    # closed 2026-11-02: a ranking is not a result).
    assert 3 in hard.evidence["ranked"] and 1 not in hard.evidence["ranked"]
    assert hard.evidence["null_verdict"] in ("not_established", "established")
    if hard.evidence["null_verdict"] == "established":
        assert 3 in hard.evidence["series"]
    assert hard.evidence["difficulty_scale"] == dissect.SCALE_RMSE
    # the defective rule, on the same frame: lowest mean R² -> series 1
    per_r2 = df.groupby("series")["r2"].mean()
    assert int(per_r2.idxmin()) == 1


def test_raw_r2_difficulty_is_labelled_as_such():
    """SA-6: a frame with no error column still works, and says it is on raw R²."""
    df = pd.DataFrame({"series": [1, 1, 2, 2, 3, 3, 4, 4], "model": ["rf", "mlp"] * 4,
                       "train_size": [500] * 8, "r2": [.1, .1, .6, .6, .62, .61, .59, .6],
                       "iteration": ["iteration_1"] * 8})
    qs = dissect.dissect_metrics(df, "t")
    assert qs and all(q.difficulty_scale == dissect.SCALE_R2_RAW for q in qs)
    assert all(q.evidence["difficulty_scale"] == dissect.SCALE_R2_RAW
               for q in qs if "difficulty_scale" in q.evidence)


def test_noise_ceiling_scale_is_used_when_replicates_are_available():
    """SA-6's preferred scale, where a dataset has the replicate structure for it."""
    rep = pd.DataFrame({"unit": [0] * 4 + [1] * 4,
                        "rep1": [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0],
                        "rep2": [1.1, 2.1, 3.1, 4.1, 3.0, 4.0, 5.0, 6.0]})
    noise = dissect.replicate_noise_rmse(rep, ["rep1", "rep2"], "unit")
    assert set(noise) == {0, 1} and noise[0] < noise[1]      # unit 1 is much noisier
    labels = np.array([0] * 30 + [1] * 30)
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, 60)
    pred = y + rng.normal(0, 0.3, 60)
    frame = dissect.neighborhoods_to_metrics_frame(labels, y, pred, min_n=10, noise_rmse=noise)
    assert "rmse_noise" in frame.columns
    ps = dissect._per_unit(frame, int(frame["train_size"].max()))
    assert ps.attrs["difficulty_scale"] == dissect.SCALE_NOISE
    # a PARTIAL noise map would silently mix two scales -> it is dropped entirely
    partial = dissect.neighborhoods_to_metrics_frame(labels, y, pred, min_n=10, noise_rmse={0: 0.1})
    assert "rmse_noise" not in partial.columns


# ======================================== ST-8: family-wise p-value and the n floor ===
def test_null_report_rate_is_nominal_where_the_old_rule_reported_most_of_the_time():
    """ST-8: at the 8 units a pooled dissection has, the old 'max |r| >= 0.4' rule reports under
    H0 roughly two times in three. The rebuilt selector reports at its nominal level."""
    n, trials = 8, 400
    rng = np.random.default_rng(11)
    new_hits = old_hits = 0
    for _ in range(trials):
        diff = pd.Series(rng.normal(size=n))
        cov = pd.DataFrame({c: rng.normal(size=n)
                            for c in ("gc", "gc_sd", "seq_entropy", "kmer_dispersion")})
        qs = dissect.correlate_difficulty(diff, cov, "null", scale=dissect.SCALE_RMSE,
                                          higher_is_harder=True, n_perm=400,
                                          seed=int(rng.integers(1 << 30)))
        new_hits += bool(qs and qs[0].evidence["reported"])
        best = max(abs(float(np.corrcoef(cov[c], diff)[0, 1])) for c in cov.columns)
        old_hits += best >= 0.4                       # the pre-rebuild gate, same draws
    assert old_hits / trials > 0.5, f"setup: the old rule should fire often (got {old_hits/trials})"
    assert new_hits / trials < 0.15, f"family-wise rate too high: {new_hits / trials}"


def test_every_reported_covariate_carries_a_family_wise_pvalue():
    """ST-8: a report is a test result, not a maximum. The p-value is over the WHOLE family."""
    n = 40
    rng = np.random.default_rng(3)
    base = rng.normal(size=n)
    cov = pd.DataFrame({"gc": base + rng.normal(0, 0.4, n),
                        "gc_sd": rng.normal(size=n), "seq_entropy": rng.normal(size=n),
                        "kmer_dispersion": rng.normal(size=n)})
    qs = dissect.correlate_difficulty(pd.Series(base), cov, "t", scale=dissect.SCALE_RMSE,
                                      higher_is_harder=True)
    ev = qs[0].evidence
    assert ev["reported"] and ev["covariate"] == "gc"
    assert ev["test"] == "max_t_step_down_permutation"
    assert 0 < ev["p_max_t"] <= ALPHA
    assert set(ev["candidate_family"]) == set(cov.columns) and ev["n_candidates"] == 4
    # the family-wise p is never SMALLER than the per-candidate one it was derived from
    assert ev["p_max_t_by_covariate"]["gc"] >= 1.0 / (ev["n_permutations"] + 1)
    assert set(ev["r_by_covariate"]) == set(cov.columns)       # every candidate is reported, not just the winner


def test_selector_refuses_below_the_measured_validity_floor():
    """ST-8: with n <= 4 the exact permutation null cannot reach p<=0.05 for ANY data, so a report
    there is unconditionally invalid. The floor is computed, not configured."""
    assert selection.min_units_for_alpha(0.05) == 5 == selection.MIN_UNITS_VALID
    assert selection.min_units_for_alpha(0.01) == 6
    perfect = pd.Series([1.0, 2.0, 3.0, 4.0])
    cov = pd.DataFrame({"gc": [1.0, 2.0, 3.0, 4.0]})           # r = +1.0 exactly
    res = selection.max_t_correlation(perfect.to_numpy(), {"gc": cov["gc"].to_numpy()})
    assert res["reported"] is False and res["refusal"] and "4 units" in res["refusal"]
    assert dissect.correlate_difficulty(perfect, cov, "t") == []


def test_small_n_reports_are_flagged_unreliable_and_large_n_are_not():
    """ST-8: below the measured reliability floor a rejection is valid but the magnitude is not
    quotable, and the question has to say so."""
    def _run(n, seed):
        rng = np.random.default_rng(seed)
        base = np.linspace(0, 1, n) + rng.normal(0, 0.03, n)
        cov = pd.DataFrame({"gc": base, "gc_sd": rng.normal(size=n),
                            "seq_entropy": rng.normal(size=n)})
        return dissect.correlate_difficulty(pd.Series(base), cov, "t",
                                            scale=dissect.SCALE_RMSE, higher_is_harder=True)[0]

    small = _run(8, 5)
    assert small.evidence["reported"] and small.evidence["selection_reliability"] == "low"
    assert "effect_size_caveat" in small.evidence
    big = _run(selection.MIN_UNITS_RELIABLE + 10, 5)
    assert big.evidence["reported"] and big.evidence["selection_reliability"] == "ok"
    assert "effect_size_caveat" not in big.evidence


def test_negative_covariate_result_is_reported_not_silently_dropped():
    """ST-8/SA-6: the dissection reports the OUTCOME of the covariate test either way, so "we
    looked and found nothing" cannot be mistaken for "we did not look"."""
    n = 30
    rng = np.random.default_rng(9)
    cov = pd.DataFrame({c: rng.normal(size=n) for c in ("gc", "gc_sd", "seq_entropy")})
    q = dissect.correlate_difficulty(pd.Series(rng.normal(size=n)), cov, "t",
                                     scale=dissect.SCALE_RMSE, higher_is_harder=True)[0]
    assert q.kind == "difficulty_covariate"
    assert q.evidence["reported"] is False and q.evidence["covariate"] is None
    assert q.evidence["best_candidate"] in cov.columns and q.evidence["p_max_t"] > ALPHA
    assert q.intervention_type == "feature_representation"     # still steers the loop


def test_step_down_max_t_is_monotone_and_handles_a_single_candidate():
    obs = {"a": 3.0, "b": 2.0, "c": 1.0}
    rng = np.random.default_rng(0)
    null = {k: rng.normal(size=500) for k in obs}
    res = selection.step_down_max_t(obs, null)
    ps = [res["adjusted"][k] for k in res["ordered"]]
    assert res["ordered"] == ["a", "b", "c"]
    assert ps == sorted(ps), "step-down adjusted p-values must be monotone in the ordering"
    one = selection.step_down_max_t({"a": 3.0}, {"a": rng.normal(size=500)})
    assert one["selected"] == "a" and one["p_max_t"] <= 0.05
    assert selection.step_down_max_t({}, {})["selected"] is None


# ================================= ML-10: train-fitted clusters, variance-matched null ===
def test_clusters_are_fitted_on_train_sequences_not_on_the_rows_they_are_scored_on():
    """ML-10: a partition chosen on the test rows can carve out a neighbourhood the training data
    never contained. Fitted on train, a novel test composition is folded into an existing cluster;
    fitted in-sample it gets a cluster of its own."""
    train = (["GCGCGCGCGCGC" * 5] * 60) + (["ATATATATATAT" * 5] * 60)
    test = (["GCGCGCGCGCGC" * 5] * 20 + ["ATATATATATAT" * 5] * 20
            + ["ACACACACACAC" * 5] * 20)            # a third composition, absent from train
    train_fitted = dissect.fit_sequence_clusters(train, k=2, seed=0).predict(test)
    assert len(set(train_fitted)) == 2                     # no cluster for the unseen composition
    assert len(set(dissect.cluster_sequences(test, k=3, seed=0))) == 3   # in-sample: it gets one
    # and the stamp says which happened
    rng = np.random.default_rng(0)
    y = rng.normal(size=60)
    pred = y + rng.normal(0, 0.5, 60)
    with_train = dissect.dissect_pooled_predictions(test, y, pred, "t", k=2, min_n=10,
                                                    train_sequences=train, n_draw=300)
    assert with_train and all(q.evidence["cluster_fit"] == "train_sequences" for q in with_train)
    assert all("cluster_fit_caveat" not in q.evidence for q in with_train)
    without = dissect.dissect_pooled_predictions(test, y, pred, "t", k=2, min_n=10, n_draw=300)
    assert without and all(q.evidence["cluster_fit"] == "test_sequences" for q in without)
    assert all("cluster_fit_caveat" in q.evidence for q in without)


def test_hardest_cluster_is_the_high_error_one_not_the_low_variance_one():
    """ML-10: the unit with the worst per-cluster R² has one sixth of the other unit's error.
    The rebuilt ranking names the high-error unit; the R² ranking would name the other."""
    train, seqs, y, pred, n = _two_group_pooled(seed=4)
    labels = dissect.fit_sequence_clusters(train, k=2, seed=0).predict(seqs)
    a_label, b_label = int(labels[0]), int(labels[-1])
    assert a_label != b_label, "setup: the two compositions must land in different clusters"
    frame = dissect.neighborhoods_to_metrics_frame(labels, y, pred, min_n=20)
    per = frame.groupby("series")[["r2", "rmse"]].mean()
    assert int(per["r2"].idxmin()) == a_label        # the defective ranking picks the low-Var unit
    assert int(per["rmse"].idxmax()) == b_label      # the error scale picks the other one

    qs = dissect.dissect_pooled_predictions(seqs, y, pred, "t", k=2, min_n=20,
                                            train_sequences=train, n_draw=1000)
    hard = [q for q in qs if q.kind == "series_difficulty"]
    assert hard, "expected a difficulty verdict"
    named = set(hard[0].evidence["series"])
    assert b_label in named and a_label not in named
    assert hard[0].evidence["test"] == "y_matched_random_partition"


def test_cluster_null_does_not_flag_units_of_a_random_partition():
    """ML-10: the null's own false-positive behaviour. Homoscedastic data cut into random parts has
    no hard neighbourhood; a ranking that divided by each part's Var(y) would still name one."""
    rng = np.random.default_rng(7)
    n = 600
    y = rng.normal(0, 1, n)
    pred = y + rng.normal(0, 0.5, n)
    flagged = 0
    reps = 25
    for rep in range(reps):
        labels = np.random.default_rng(rep).integers(0, 6, n)
        res = dissect.y_matched_partition_null(labels, y, pred, n_draw=400, seed=rep)
        assert len(res["units"]) == 6
        flagged += bool(res["harder_than_null"])
    assert flagged / reps <= 0.2, f"null flagged a unit in {flagged}/{reps} random partitions"
    # a ranking on per-unit R² always produces a "hardest" unit, however random the partition
    labels = np.random.default_rng(0).integers(0, 6, n)
    frame = dissect.neighborhoods_to_metrics_frame(labels, y, pred, min_n=20)
    assert frame.groupby("series")["r2"].mean().idxmin() is not None


def test_cluster_null_reports_the_variance_it_actually_matched():
    """ML-10: the null is matched at y-quantile-bin resolution, so the match is a measurement, not
    a promise — each unit carries the realised null/observed Var(y) ratio."""
    rng = np.random.default_rng(2)
    n = 400
    y = rng.normal(0, 1, n)
    pred = y + rng.normal(0, 0.5, n)
    labels = rng.integers(0, 5, n)
    res = dissect.y_matched_partition_null(labels, y, pred, n_draw=500, seed=0)
    for u, row in res["units"].items():
        assert row["var_ratio_null_over_observed"] is not None
        assert row["variance_match"] in ("ok", "loose_conservative", "loose_liberal")
        assert 0.5 < row["var_ratio_null_over_observed"] < 2.0, (u, row)
    assert "variance_match_loose" in res


def test_hard_for_every_model_requires_two_model_families():
    """ML-10: a single-family dissection cannot claim a model-agnostic ceiling."""
    rows_two, rows_one = [], []
    for s, (rf, mlp) in {1: (1.4, 1.42), 2: (1.38, 1.41), 3: (0.3, 0.31),
                         4: (0.32, 0.30), 5: (0.29, 0.33)}.items():
        rows_two += [(s, "rf", 500, 0.1, rf), (s, "mlp", 500, 0.1, mlp)]
        rows_one += [(s, "rf", 500, 0.1, rf)]
    two = dissect.dissect_metrics(_frame(rows_two), "t")
    one = dissect.dissect_metrics(_frame(rows_one), "t")
    assert "model_agnostic_ceiling" in {q.kind for q in two}
    assert "model_agnostic_ceiling" not in {q.kind for q in one}
    weak = next(q for q in one if q.kind == "single_model_weak")
    assert weak.evidence["n_model_families"] == 1 and weak.evidence["model_families"] == ["rf"]
    assert weak.intervention_type == "model_architecture"      # "go run a second family"
    # and no single-family question claims anything about every model
    assert not [q for q in one if "every model" in q.observation.lower()
                and "cannot support" not in q.observation]


def test_pooled_probe_returns_more_than_one_family_and_the_train_sequences():
    """ML-10: the two inputs the pre-rebuild probe could not supply. Checked on a stub runner so
    the test needs no data."""
    from seq2yield.experiments import pooled_runner as P
    tr = pd.DataFrame({"Sequence": ["ACGT"] * 10, "Protein": np.arange(10.0)})
    te = pd.DataFrame({"Sequence": ["TGCA"] * 4, "Protein": np.arange(4.0)})
    calls = []

    def _fit_predict(spec, fam, train, test, **kw):
        calls.append(fam)
        return np.zeros(len(test)) + (1.0 if fam == "rf" else 2.0)

    saved = (P.holdout, P.subsample, P.fit_predict)
    P.holdout = lambda spec, frac=0.1: (tr, te)
    P.subsample = lambda train, size, policy, seed: train
    P.fit_predict = _fit_predict
    try:
        probe = dissect.pooled_probe("x", families=("rf", "mlp"))
        legacy = dissect.pooled_predictions("x", model_family="rf")
    finally:
        P.holdout, P.subsample, P.fit_predict = saved
    assert probe["families"] == ["rf", "mlp"] and calls[:2] == ["rf", "mlp"]
    assert probe["train_sequences"] == ["ACGT"] * 10
    assert len(probe["test_sequences"]) == 4 and len(probe["y_true"]) == 4
    assert len(legacy) == 3 and list(legacy[2]) == [1.0] * 4     # legacy 3-tuple still works


# ============================================================ consumers and storage ===
def test_pre_rebuild_insight_files_load_flagged_instead_of_crashing(tmp_path, monkeypatch):
    """A question written before the rebuild has no difficulty_scale; it must still load, and must
    carry the fact that its difficulty is raw R² and its covariate has no p-value."""
    legacy = ('{"id":"q_difficulty_covariate_abc","dataset":"ecoli","kind":"difficulty_covariate",'
              '"observation":"Per-series R2 correlates with \'expr_std\' (r=+0.53)",'
              '"hypothesis":"h","suggested_intervention":"stratify by the correlated covariate",'
              '"evidence":{"covariate":"expr_std","pearson_r":0.533,"n_series":56},'
              '"priority":0.76}')
    p = tmp_path / "ecoli.jsonl"
    p.write_text(legacy + "\n{not json}\n", encoding="utf-8")
    monkeypatch.setattr(dissect, "_insight_path", lambda d: p)
    qs = dissect.load_questions("ecoli")
    assert len(qs) == 1                                        # the broken line is skipped
    assert qs[0].pre_rebuild and qs[0].difficulty_scale == "r2_raw_legacy"
    assert qs[0].evidence["pre_rebuild"] is True
    assert "p-value" in qs[0].evidence["pre_rebuild_note"]
    # a REBUILT question round-trips with its scale intact and is not flagged
    fresh = dissect.correlate_difficulty(
        pd.Series(np.linspace(0, 1, 30)),
        pd.DataFrame({"gc": np.linspace(0, 1, 30) + np.random.default_rng(0).normal(0, .1, 30)}),
        "ecoli", scale=dissect.SCALE_RMSE, higher_is_harder=True)
    dissect.save_questions("ecoli", fresh)
    back = dissect.load_questions("ecoli")
    assert [q.id for q in back] == [q.id for q in fresh]
    assert back[0].difficulty_scale == dissect.SCALE_RMSE and not back[0].pre_rebuild


def test_phase_controller_still_reads_the_rebuilt_questions(monkeypatch):
    """The GeneratedQuestion contract phase.py / run_insight.py / the claim firewall consume."""
    train, seqs, y, pred, n = _two_group_pooled(seed=6)
    qs = dissect.dissect_pooled_predictions(seqs, y, pred, "yeast", k=2, min_n=20,
                                            train_sequences=train, n_draw=500)
    assert qs
    assert all(isinstance(q, dissect.GeneratedQuestion) for q in qs)
    assert qs == sorted(qs, key=lambda q: q.priority, reverse=True)
    assert dissect.to_focus_hints(qs)                           # still produces PI hints
    monkeypatch.setattr(phase.dissect, "hints_for_dataset", lambda d: ([], qs))
    ph = phase.dataset_phase("yeast", [])
    assert ph["phase"] == "neighborhood"
    assert all(isinstance(x, int) for x in ph["neighborhoods"])
    addressed = [{"dataset": "yeast", "intervention_type": t}
                 for t in {q.intervention_type for q in qs} if t]
    assert phase.dataset_phase("yeast", addressed)["phase"] == "global"
