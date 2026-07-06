"""The human-accept gate for expensive experiments: suggesting enqueues (never runs), accept/reject
update status, dispatch runs ONLY accepted items, a non-accepted dispatch is refused, and errors
don't wedge the queue. The real module call is stubbed so the test is fast + deterministic.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agents import experiment_queue as Q  # noqa: E402


@pytest.fixture
def qpath(tmp_path, monkeypatch):
    p = tmp_path / "queue.jsonl"
    # keep trace logging out of the repo during tests
    monkeypatch.setattr(Q.trace, "log_event", lambda *a, **k: None)
    return p


def test_suggest_enqueues_pending_and_does_not_run(qpath, monkeypatch):
    ran = {"n": 0}
    monkeypatch.setattr(Q, "_run", lambda kind, params: ran.__setitem__("n", ran["n"] + 1) or {})
    rec = Q.suggest("tournament", {"dataset": "sample_2019"}, "cnn vs flat", path=qpath)
    assert rec["status"] == "pending" and rec["estimated_cost_s"] > 0 and ran["n"] == 0
    assert len(Q.list_queue(path=qpath)) == 1
    with pytest.raises(ValueError):
        Q.suggest("nonsense", {}, "x", path=qpath)


def test_accept_then_dispatch_runs_and_marks_done(qpath, monkeypatch):
    monkeypatch.setattr(Q, "_run", lambda kind, params: {"winner": "cnn", "winner_significant": True})
    rec = Q.suggest("tournament", {"dataset": "sample_2019"}, "r", path=qpath)
    Q.accept(rec["id"], path=qpath)
    assert Q.list_queue("accepted", path=qpath)[0]["id"] == rec["id"]
    out = Q.dispatch(rec["id"], path=qpath)
    assert out["status"] == "done" and out["result"]["winner"] == "cnn"


def test_dispatch_refuses_unaccepted(qpath, monkeypatch):
    monkeypatch.setattr(Q, "_run", lambda kind, params: {})
    rec = Q.suggest("config_transfer", {"model": "rf"}, "r", path=qpath)
    with pytest.raises(PermissionError):
        Q.dispatch(rec["id"], path=qpath)              # still pending -> must not run
    Q.reject(rec["id"], path=qpath)
    with pytest.raises(PermissionError):
        Q.dispatch(rec["id"], path=qpath)              # rejected -> must not run


def test_run_accepted_only_runs_accepted(qpath, monkeypatch):
    monkeypatch.setattr(Q, "_run", lambda kind, params: {"ok": True})
    a = Q.suggest("tournament", {"dataset": "d"}, "r", path=qpath)
    Q.suggest("tournament", {"dataset": "d2"}, "r", path=qpath)   # left pending
    Q.accept(a["id"], path=qpath)
    done = Q.run_accepted(path=qpath)
    assert len(done) == 1 and done[0]["id"] == a["id"] and done[0]["status"] == "done"
    assert len(Q.list_queue("pending", path=qpath)) == 1          # the other is untouched


def test_dispatch_error_does_not_wedge_the_queue(qpath, monkeypatch):
    def _boom(kind, params):
        raise RuntimeError("bad params")

    monkeypatch.setattr(Q, "_run", _boom)
    rec = Q.suggest("hpo_distribution", {"dataset": "ecoli"}, "r", path=qpath)
    Q.accept(rec["id"], path=qpath)
    out = Q.dispatch(rec["id"], path=qpath)
    assert out["status"] == "error" and "bad params" in out["result"]["error"]


def test_cost_estimate_scales_with_family_and_model(qpath):
    cheap = Q.estimate_cost_s("tournament", {"family": ["ridge", "rf"], "model": "rf"})
    dear = Q.estimate_cost_s("tournament", {"family": ["ridge", "rf", "mlp", "cnn"], "model": "cnn"})
    assert dear > cheap > 0
