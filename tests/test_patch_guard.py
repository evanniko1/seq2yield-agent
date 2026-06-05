"""Patch loop tests (deterministic): protected-file guard + patch_manager apply/revert."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agents.schemas import FileOperation, PatchPlan  # noqa: E402
from orchestration import git_guard, patch_manager  # noqa: E402

ALLOWED = ["configs/model/", "src/seq2yield/models/"]


def test_guard_blocks_strict_protected():
    res = git_guard.check_paths(["src/seq2yield/training/metrics.py"], allowed_files=ALLOWED)
    assert not res["passed"]
    assert res["by_path"]["src/seq2yield/training/metrics.py"]["class"] == "strict"


def test_guard_blocks_split_and_data():
    res = git_guard.check_paths(["data/splits/splits_manifest.json",
                                 "configs/objective.yaml"], allowed_files=ALLOWED)
    assert not res["passed"] and len(res["violations"]) == 2


def test_guard_allows_freely_modifiable_within_allowed():
    res = git_guard.check_paths(["configs/model/cnn_deep.yaml"], allowed_files=ALLOWED)
    assert res["passed"]


def test_guard_blocks_freely_modifiable_outside_allowed():
    # reporting/ is freely_modifiable but not in this RunSpec's allowed_files
    res = git_guard.check_paths(["src/seq2yield/reporting/plots.py"], allowed_files=ALLOWED)
    assert not res["passed"]


def test_patch_apply_and_revert_roundtrip():
    rel = "configs/model/_pytest_tmp_variant.yaml"
    target = ROOT / rel
    assert not target.exists()
    plan = PatchPlan(proposal_id="t", run_id="t", summary="tmp",
                     operations=[FileOperation(op="create", path=rel, content="base_model: cnn\n")])
    undo = patch_manager.apply(plan)
    try:
        assert target.exists() and "cnn" in target.read_text()
    finally:
        patch_manager.revert(undo)
    assert not target.exists()        # revert deletes a created file


def test_patch_modify_requires_unique_anchor(tmp_path):
    rel = "configs/model/_pytest_tmp_anchor.yaml"
    target = ROOT / rel
    create = PatchPlan(proposal_id="t", run_id="t", summary="c",
                       operations=[FileOperation(op="create", path=rel, content="x\nx\n")])
    undo = patch_manager.apply(create)
    try:
        bad = PatchPlan(proposal_id="t", run_id="t", summary="m",
                        operations=[FileOperation(op="modify", path=rel, find="x", content="y")])
        with pytest.raises(patch_manager.PatchError):
            patch_manager.apply(bad)
    finally:
        patch_manager.revert(undo)
