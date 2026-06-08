"""C8/S3: reviewer/chair prompts carry an anchored rubric so scores discriminate
instead of clustering at 4 and the chair stops rubber-stamping overall+bonus."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agents import prompting  # noqa: E402

_PROPOSAL = {"proposal_id": "p1", "intervention_type": "feature_representation",
             "model_family": "mlp", "comparator_model": "mlp", "feature_set": "kmer"}


def test_reviewer_prompt_has_anchored_rubric():
    sysmsg, user = prompting.reviewer_prompt("methodology_reviewer", _PROPOSAL)
    # full-range anchoring + explicit default-of-3 to break the cluster-at-4 failure mode
    assert "FULL 1-5 range" in user and "default is 3" in user
    # each scored dimension is anchored
    for dim in ("feasibility", "scientific_value", "confoundedness", "reproducibility"):
        assert dim in user
    # same-model-baseline confound rule for non-architecture studies is surfaced
    assert "same model_family" in user
    # must name one concrete fix, not generic praise
    assert "SINGLE most important concrete fix" in user


def test_chair_prompt_requires_justified_choice():
    sysmsg, user = prompting.chair_prompt([_PROPOSAL], {"p1": {"overall": 4.0, "sound": True}})
    # deterministic rule preserved
    assert "HIGHEST 'overall'" in user
    # tie-break + non-trivial rationale (runner-up + required ablation)
    assert "ties" in user and "runner-up" in user and "required_ablation" in user
