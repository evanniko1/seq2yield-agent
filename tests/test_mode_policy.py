"""--auto fast-vs-full mode policy: deterministic over coverage + phase + prior provisional signal +
budget. Monkeypatches coverage/phase so each branch is exercised in isolation."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agents import mode_policy  # noqa: E402


def _patch(monkeypatch, *, status="untested", phase="global"):
    monkeypatch.setattr(mode_policy.question_space, "record_cell_id", lambda r: "C")
    monkeypatch.setattr(mode_policy.question_space, "coverage", lambda recs: {"C": {"status": status}})
    monkeypatch.setattr(mode_policy, "dataset_phase",
                        lambda d, recs: {"phase": phase, "neighborhoods": [1] if phase == "neighborhood" else []})


def test_promising_provisional_escalates_to_full(monkeypatch):
    # even in neighborhood phase, a promising provisional probe -> confirm (breaks the probe loop)
    _patch(monkeypatch, status="untested", phase="neighborhood")
    mode, why = mode_policy.decide("C", "ecoli", [{"provisional": True, "mean_delta": 0.05}])
    assert mode == "full" and "confirm" in why


def test_inconclusive_cell_goes_full(monkeypatch):
    _patch(monkeypatch, status="inconclusive", phase="neighborhood")
    mode, why = mode_policy.decide("C", "ecoli", [])
    assert mode == "full" and "inconclusive" in why


def test_neighborhood_phase_triggers_fast_probe(monkeypatch):
    _patch(monkeypatch, status="untested", phase="neighborhood")
    mode, why = mode_policy.decide("C", "ecoli", [])
    assert mode == "fast" and "neighborhood" in why


def test_broad_frontier_or_tight_budget_triages_fast(monkeypatch):
    _patch(monkeypatch, status="untested", phase="global")
    assert mode_policy.decide("C", "ecoli", [], uncovered_frac=0.8)[0] == "fast"     # broad frontier
    assert mode_policy.decide("C", "ecoli", [], budget_tight=True)[0] == "fast"       # tight budget


def test_default_is_a_full_rigorous_run(monkeypatch):
    _patch(monkeypatch, status="untested", phase="global")
    mode, why = mode_policy.decide("C", "ecoli", [], uncovered_frac=0.1, budget_tight=False)
    assert mode == "full" and "default" in why


def test_weak_provisional_does_not_escalate(monkeypatch):
    # a provisional probe below the promising threshold must NOT trigger a confirmation escalation
    _patch(monkeypatch, status="untested", phase="global")
    mode, why = mode_policy.decide("C", "ecoli", [{"provisional": True, "mean_delta": 0.001}],
                                   uncovered_frac=0.1)
    assert mode == "full" and "confirm" not in why          # falls through to default, not escalation
