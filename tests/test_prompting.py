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
    p = prompting.reviewer_prompt("methodology_reviewer", _PROPOSAL)
    user = p.user
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
    p = prompting.chair_prompt([_PROPOSAL], {"p1": {"overall": 4.0, "sound": True}})
    user = p.user
    # deterministic rule preserved
    assert "HIGHEST 'overall'" in user
    # tie-break + non-trivial rationale (runner-up + required ablation)
    assert "ties" in user and "runner-up" in user and "required_ablation" in user


# ---- C11: versioned templates carried + recorded ----
def test_prompts_carry_template_and_version():
    for builder, name in ((prompting.reviewer_prompt("methodology_reviewer", _PROPOSAL), "reviewer"),
                          (prompting.chair_prompt([_PROPOSAL], {"p1": {"overall": 1.0, "sound": True}}), "chair"),
                          (prompting.generator_prompt(2), "generator")):
        assert builder.template == name
        assert builder.version == prompting.TEMPLATE_VERSIONS[name]
    m = prompting.meta("reviewer")
    assert m == {"prompt_template": "reviewer", "prompt_version": prompting.TEMPLATE_VERSIONS["reviewer"]}


# ---- C12: JSON blobs are trimmed (empty fields dropped, fields selected) ----
def test_compact_json_strips_empty_and_selects_fields():
    noisy = {"proposal_id": "p1", "model_family": "mlp", "comparator_model": "mlp",
             "feature_set": "kmer", "required_controls": [], "expected_failure_modes": None,
             "notes": "", "scope": "global", "param_count": 0, "ci_excludes_zero": False}
    compact = prompting.compact_json(prompting._select(noisy, prompting._REVIEW_FIELDS), indent=2)
    assert "required_controls" not in compact      # empty list dropped
    assert "notes" not in compact                  # not in whitelist
    assert "model_family" in compact and "kmer" in compact
    # full pretty dump is materially larger than the trimmed one (token win)
    import json
    full = json.dumps(noisy, indent=2)
    assert len(compact) < len(full)
    # zeros/False are real signal and must survive _strip_empty
    kept = prompting._strip_empty({"a": 0, "b": False, "c": None, "d": []})
    assert kept == {"a": 0, "b": False}
