"""Tests for the protected-file guard (orchestration/git_guard.py)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orchestration import git_guard  # noqa: E402


def test_classify():
    assert git_guard.classify("data/splits/iter1.csv") == "strict"
    assert git_guard.classify("src/seq2yield/training/metrics.py") == "strict"
    assert git_guard.classify("configs/metrics.yaml") == "conditional"
    assert git_guard.classify("src/seq2yield/statistics/bootstrap.py") == "conditional"
    assert git_guard.classify("src/seq2yield/models/cnn.py") == "freely_modifiable"
    assert git_guard.classify("some/random/file.py") == "require_review"


def test_strict_always_fails():
    res = git_guard.check_paths(["data/splits/x.csv"])
    assert not res["passed"] and "data/splits/x.csv" in res["violations"]


def test_conditional_needs_human_review():
    assert not git_guard.check_paths(["configs/metrics.yaml"])["passed"]
    assert git_guard.check_paths(["configs/metrics.yaml"], human_review=True)["passed"]


def test_freely_modifiable_within_allowed():
    p = "src/seq2yield/models/cnn.py"
    assert git_guard.check_paths([p], allowed_files=["src/seq2yield/models/*"])["passed"]
    # freely-modifiable but outside the RunSpec allow-list -> fail
    assert not git_guard.check_paths([p], allowed_files=["src/seq2yield/features/*"])["passed"]


def test_empty_changeset_passes():
    assert git_guard.check_paths([])["passed"]
