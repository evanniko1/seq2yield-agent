"""Code provenance + baseline-registry invalidation (audit ML-02).

The failure this guards is not a crash: the registry stayed readable, the comparisons kept
returning verdicts, and an unchanged RF measured −0.0158 R² against its own stored baseline
because one sklearn default had moved underneath it. Nothing could notice, because nothing
recorded what the stored numbers were produced with. These tests pin the record and the rule that
reads it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orchestration import baseline_guard as BG  # noqa: E402
from seq2yield.training import provenance as P  # noqa: E402


def _fake_repo(tmp_path: Path, *, numerics="x = 1\n", docs="hello\n") -> Path:
    """A miniature checkout with one value-deciding file and one that decides nothing."""
    (tmp_path / "src/seq2yield/models").mkdir(parents=True)
    (tmp_path / "src/seq2yield/models/classical.py").write_text(numerics, encoding="utf-8")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/NOTES.md").write_text(docs, encoding="utf-8")
    return tmp_path


# --------------------------------------------------------------- what gets recorded
def test_numerics_fingerprint_sees_model_code_and_ignores_prose():
    import tempfile
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b, \
            tempfile.TemporaryDirectory() as c:
        base = P.numerics_fingerprint(str(_fake_repo(Path(a))))["fingerprint"]
        same_code_new_docs = P.numerics_fingerprint(
            str(_fake_repo(Path(b), docs="rewritten\n")))["fingerprint"]
        changed_code = P.numerics_fingerprint(
            str(_fake_repo(Path(c), numerics="x = 2\n")))["fingerprint"]
    assert base == same_code_new_docs      # a docs commit must not invalidate a registry
    assert base != changed_code            # a model-code change must


def test_numerics_fingerprint_is_line_ending_insensitive():
    """Otherwise a Windows checkout and a Linux checkout of the same commit would disagree and
    every registry would look stale on one of them."""
    import tempfile
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        lf = _fake_repo(Path(a), numerics="def f():\n    return 1\n")
        crlf = _fake_repo(Path(b))
        (crlf / "src/seq2yield/models/classical.py").write_bytes(b"def f():\r\n    return 1\r\n")
        assert (P.numerics_fingerprint(str(lf))["fingerprint"]
                == P.numerics_fingerprint(str(crlf))["fingerprint"])


def test_the_real_checkout_fingerprints_its_value_deciding_files():
    f = P.numerics_fingerprint()
    assert f["n_files"] >= 10 and len(f["fingerprint"]) == 16
    assert f == P.numerics_fingerprint()                      # stable within a checkout


def test_model_defaults_record_the_knobs_in_full_not_just_a_hash():
    d = P.model_defaults()
    assert "max_features" in d["rf"]["params"]                # the ML-02 culprit is visible
    assert d["rf"]["hash"] and d["mlp"]["hash"]
    assert d["rf"]["hash"] != d["mlp"]["hash"]


def test_diff_defaults_names_the_drifted_knob():
    """The whole value of recording params: "the fingerprint changed" is a shrug."""
    old = {"rf": {"hash": "aaa", "params": {"max_features": 1.0, "n_estimators": 100}}}
    new = {"rf": {"hash": "bbb", "params": {"max_features": "sqrt", "n_estimators": 100}}}
    assert P.diff_defaults(old, new) == ["rf.max_features: 1.0 -> 'sqrt'"]
    assert P.diff_defaults(old, old) == []
    assert P.diff_defaults(None, new) == []                   # nothing recorded -> nothing to say


def test_environment_carries_the_provenance_stamp():
    from seq2yield.training.reproducibility import environment
    env = environment()
    assert set(env) >= {"python", "packages", "code", "numerics", "model_defaults"}
    assert env["code"]["git_sha"] is None or len(env["code"]["git_sha"]) == 40
    assert env["numerics"]["fingerprint"]


def test_a_provenance_failure_cannot_sink_a_run(monkeypatch):
    from seq2yield.training import reproducibility as R
    monkeypatch.setattr(P, "stamp", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    env = R.environment()
    assert "provenance_error" in env and "python" in env      # degraded, not fatal


# --------------------------------------------------------------- the invalidation rule
def _registry(tmp_path: Path, run_id: str, models=("cnn", "rf"), *, env: dict | None = None,
              dataset_hash="d" * 16, split_hash="s" * 16, series=(1, 2, 3), iterations=(1, 2),
              sizes=(500, 2000), rows_series=None) -> Path:
    """A miniature registry: run-cards declaring the intended grid + a metrics table.
    `rows_series` lets a test write FEWER series than the cards declare (a truncated registry)."""
    rdir = tmp_path / "experiments/runs" / run_id
    for m in models:
        (rdir / m).mkdir(parents=True, exist_ok=True)
        (rdir / m / "run_card.json").write_text(json.dumps(
            {"run_id": f"{run_id}-{m}", "model_family": m, "dataset_hash": dataset_hash,
             "split_hash": split_hash, "n_series": len(series), "iterations": list(iterations),
             "results": {str(s): {"r2_mean": 0.5} for s in sizes},
             "environment": env if env is not None else {}}), encoding="utf-8")
    rows = ["iteration,series,model,train_size,r2"]
    for i in iterations:
        for s in (rows_series if rows_series is not None else series):
            for m in models:
                for size in sizes:
                    rows.append(f"iteration_{i},{s},{m},{size},0.5")
    (rdir / "metrics.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return tmp_path


def _stamped(fingerprint: str, defaults: dict | None = None) -> dict:
    return {"packages": {}, "code": {"git_sha": "f" * 40},
            "numerics": {"fingerprint": fingerprint, "n_files": 24},
            "model_defaults": defaults or P.model_defaults()}


def test_a_registry_with_no_run_card_is_refused(tmp_path):
    out = BG.check("ghost", ["cnn", "rf"], root=tmp_path)
    assert out["ok"] is False and out["violations"] == ["missing_run_card"]


def test_an_unstamped_registry_is_refused(tmp_path):
    """The pre-2026-11-01 case: readable numbers, no record of what made them."""
    _registry(tmp_path, "old")
    out = BG.check("old", ["cnn", "rf"], root=tmp_path)
    assert out["ok"] is False and out["violations"] == ["no_provenance"]
    assert "predates provenance stamping" in out["reason"]


def test_a_stale_registry_is_refused_and_the_drift_is_named(tmp_path):
    drifted = {"rf": {"hash": "old", "params": {"max_features": 1.0}}}
    _registry(tmp_path, "stale", env=_stamped("0" * 16, drifted))
    out = BG.check("stale", ["rf"], root=tmp_path)
    assert out["ok"] is False and "numerics_fingerprint" in out["violations"]
    assert any("rf.max_features: 1.0 ->" in d for d in out["drift"])


def test_a_matching_registry_is_accepted(tmp_path):
    current = P.numerics_fingerprint()["fingerprint"]
    _registry(tmp_path, "fresh", env=_stamped(current))
    out = BG.check("fresh", ["cnn", "rf"], root=tmp_path)
    assert out["ok"] is True and out["violations"] == []
    assert current[:8] in out["reason"]


def test_a_different_git_sha_alone_does_not_refuse(tmp_path):
    """Deliberate: a guard that fires on every commit gets switched off. Only the value-deciding
    files, the dataset and the splits can invalidate a stored result."""
    _registry(tmp_path, "fresh", env=_stamped(P.numerics_fingerprint()["fingerprint"]))
    assert BG.config().get("require", {}).get("git_sha") is False
    assert BG.check("fresh", ["rf"], root=tmp_path)["ok"] is True


def test_a_truncated_registry_is_refused(tmp_path):
    """The builder checkpoints after every series, so a registry under construction is a readable
    file covering an arbitrary SUBSET — and a subset is not a neutral baseline (the first ten
    series are easier than average, audit SA-2). Completeness is checked against the run-cards."""
    fp = P.numerics_fingerprint()["fingerprint"]
    _registry(tmp_path, "partial", env=_stamped(fp), series=(1, 2, 3), rows_series=(1,))
    out = BG.check("partial", ["cnn", "rf"], root=tmp_path)
    assert out["ok"] is False and out["violations"] == ["incomplete_registry"]
    assert "1/3 series" in out["reason"]
    assert BG.registry_status("partial", root=tmp_path)["ready"] is False


def test_an_unfinished_registry_has_no_run_cards_so_it_is_refused(tmp_path):
    """The run-cards are written last, which is what makes a half-built registry detectable."""
    rdir = tmp_path / "experiments/runs/building"
    rdir.mkdir(parents=True)
    (rdir / "metrics.csv").write_text("iteration,series,model,train_size,r2\n", encoding="utf-8")
    assert BG.registry_status("building", root=tmp_path)["ready"] is False
    assert BG.check("building", ["rf"], root=tmp_path)["violations"] == ["missing_run_card"]


def test_a_complete_registry_reports_its_grid(tmp_path):
    _registry(tmp_path, "whole", env=_stamped(P.numerics_fingerprint()["fingerprint"]))
    st = BG.registry_status("whole", root=tmp_path)
    assert st["ready"] is True and st["n_rows"] == 2 * 3 * 2 * 2
    assert st["rf"]["n_series"] == 3 and st["rf"]["train_sizes"] == [500, 2000]


def test_no_registry_configured_is_refused_not_assumed(tmp_path):
    out = BG.check(None, ["rf"], root=tmp_path)
    assert out["ok"] is False and out["violations"] == ["missing_run_id"]


def test_there_is_exactly_one_pointer_to_the_registry():
    from agents import council
    assert BG.registry_run_id() and council.BASELINE_RUN_ID == BG.registry_run_id()


def test_the_superseded_registry_is_still_refused_by_the_live_guard():
    """The audit finding itself, as a test: the registry every pre-2026-11-01 claim was compared
    against cannot be used as a control by today's code."""
    old = ROOT / "experiments/runs/2026-06-04-full56"
    if not old.exists():
        return
    out = BG.check("2026-06-04-full56", ["cnn", "rf"])
    assert out["ok"] is False and out["violations"] == ["no_provenance"]


# --------------------------------------------------------------- the harness acts on it
def test_the_harness_falls_back_to_an_in_run_baseline_when_the_registry_is_refused(monkeypatch):
    from orchestration import execution_harness as EH
    from seq2yield.experiments.run_spec import AcceptancePolicy, RunSpec
    spec = RunSpec(run_id="t", dataset="ecoli", intervention_type="model_architecture",
                   model_family="cnn", train_sizes=[500],
                   acceptance_policy=AcceptancePolicy(track="performance",
                                                      baseline_run_id="2026-06-04-full56",
                                                      baseline_model="rf"))
    use, prov = EH._baseline_choice(spec, spec.acceptance_policy)
    assert use is False and prov is not None and prov["violations"] == ["no_provenance"]

    monkeypatch.setattr(EH.baseline_guard, "check",
                        lambda rid, models, **k: {"ok": True, "reason": "ok", "violations": [],
                                                  "drift": [], "registry_run_id": rid})
    assert EH._baseline_choice(spec, spec.acceptance_policy)[0] is True


def test_a_confounded_comparison_never_reaches_the_registry_at_all(monkeypatch):
    """A non-plain axis (here: k-mer features) is confounded against the registry regardless of
    provenance, so it must not even consult it — it always trains its own control."""
    from orchestration import execution_harness as EH
    from seq2yield.experiments.run_spec import AcceptancePolicy, RunSpec
    monkeypatch.setattr(EH.baseline_guard, "check",
                        lambda *a, **k: {"ok": True, "violations": [], "drift": [],
                                         "reason": "", "registry_run_id": "x"})
    spec = RunSpec(run_id="t", dataset="ecoli", intervention_type="feature_representation",
                   model_family="cnn", feature_set="kmer", train_sizes=[500],
                   acceptance_policy=AcceptancePolicy(track="performance", baseline_model="cnn"))
    assert EH._baseline_choice(spec, spec.acceptance_policy) == (False, None)
