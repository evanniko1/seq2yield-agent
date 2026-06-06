"""Test the read-only dashboard export (pure HTML builder)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seq2yield.reporting.dashboard_export import build_html  # noqa: E402


def test_build_html_contains_runs_claims_and_coverage():
    records = [
        {"run_id": "r1", "candidate_model": "cnn", "baseline_model": "rf",
         "intervention_type": "model_architecture", "status": "accepted", "mean_delta": 0.03,
         "claim_allowed": "cnn beats rf"},
        {"run_id": "r2", "candidate_model": "rf", "baseline_model": "rf",
         "intervention_type": "training_procedure", "status": "inconclusive", "mean_delta": -0.002},
    ]
    claims = [{"run_id": "r1", "claim": "cnn beats rf"}]
    html = build_html(records, claims)
    assert "<html" in html and "research dashboard" in html
    assert "cnn beats rf" in html                     # claim shown
    assert "model_architecture" in html and "training_procedure" in html
    assert "question-space coverage" in html.lower()
    assert "accepted" in html and "inconclusive" in html


def test_build_html_handles_empty():
    html = build_html([], [])
    assert "<html" in html and "no accepted claims yet" in html
