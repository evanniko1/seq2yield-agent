"""C9: exercise the human-review gate for conditional-protected changes end-to-end.

The protected-file guard's classification was already unit-tested (test_patch_guard.py);
this drives the *decision + audit* layer that authorizes a conditional change, and the
safety invariants:
  - conditional path + no approver           -> DENY (run halts awaiting review)
  - conditional path + named approver        -> GRANT (harness guard then passes)
  - strict path (even with an approver)      -> DENY (never approvable)
  - freely-modifiable only                   -> no review required
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orchestration import approvals, git_guard  # noqa: E402

CONDITIONAL = "src/seq2yield/experiments/compare.py"     # conditional in protected_files.yaml
STRICT = "src/seq2yield/training/metrics.py"             # strict — never agent-modifiable
FREE = "src/seq2yield/models/cnn.py"                     # freely-modifiable


def test_conditional_without_approver_is_denied():
    d = approvals.decide("r1", [CONDITIONAL], approver=None)
    assert d.granted is False
    assert CONDITIONAL in d.conditional_paths and not d.strict_paths
    assert "HALT" in d.reason


def test_conditional_with_named_approver_is_granted():
    d = approvals.decide("r2", [CONDITIONAL], approver="dr_researcher")
    assert d.granted is True and d.approver == "dr_researcher"
    assert CONDITIONAL in d.conditional_paths
    # and the harness guard now accepts it when human_review is carried through
    g = git_guard.check_paths([CONDITIONAL], human_review=True)
    assert g["passed"] is True


def test_strict_is_never_approvable_even_with_approver():
    d = approvals.decide("r3", [STRICT], approver="dr_researcher")
    assert d.granted is False                       # invariant: strict can't be authorized
    assert STRICT in d.strict_paths
    # belt-and-suspenders: the guard ALSO refuses strict even if human_review is forced True
    g = git_guard.check_paths([STRICT], human_review=True)
    assert g["passed"] is False
    assert g["by_path"][STRICT]["class"] == "strict"


def test_mixed_strict_plus_conditional_is_denied_by_strict():
    d = approvals.decide("r4", [CONDITIONAL, STRICT], approver="dr_researcher")
    assert d.granted is False and d.strict_paths == [STRICT]


def test_freely_modifiable_needs_no_review():
    d = approvals.decide("r5", [FREE], approver=None)
    assert d.granted is True and not d.conditional_paths and not d.strict_paths
    assert "not required" in d.reason


def test_decision_is_logged_as_artifact_and_audit_event(tmp_path):
    d = approvals.decide("r6", [CONDITIONAL], approver="dr_researcher")
    approvals.log(tmp_path, d)
    art = json.loads((tmp_path / "approval_decision.json").read_text(encoding="utf-8"))
    assert art["granted"] is True and art["approver"] == "dr_researcher"
    audit = (tmp_path / "audit_log.jsonl").read_text(encoding="utf-8").strip().splitlines()
    rec = json.loads(audit[-1])
    assert rec["event"] == "human_review_gate" and rec["run_id"] == "r6"


# ------------------------------------------------- AG-05 / AG-06 (2026-11-02): a record is evidence
def test_a_mangled_path_cannot_be_approved():
    """The ST-3 record written on 2026-11-02 had the first character eaten off 14 of its 15 paths
    (`rc/seq2yield/...` for `src/...`) because the list went through a shell. `git_guard.classify`
    returned `require_review` for every mangled string, so the record certified review of files
    that do not exist and recorded NO approval for the two conditional-protected files the change
    actually touched — and it sat on disk as evidence for a day."""
    from orchestration import approvals
    d = approvals.decide("t", ["rc/seq2yield/experiments/compare.py", "ocs/DECISIONS.md"],
                         approver="Someone")
    assert d.granted is False
    assert "do not name anything in this worktree" in d.reason
    assert "AG-05/AG-06" in d.reason


def test_the_real_paths_those_were_meant_to_be_are_approvable():
    """The same change, spelled correctly: granted, and `compare.py` / `bootstrap.py` classify as
    CONDITIONAL rather than the generic `require_review` a mangled path fell back to."""
    from orchestration import approvals
    paths = ["src/seq2yield/experiments/compare.py", "src/seq2yield/statistics/bootstrap.py",
             "docs/DECISIONS.md"]
    d = approvals.decide("t", paths, approver="Someone")
    assert d.granted is True
    conditional = [p for p, k in d.path_classes.items() if k == "conditional"]
    assert set(conditional) == {"src/seq2yield/experiments/compare.py",
                                "src/seq2yield/statistics/bootstrap.py"}


def test_a_file_the_patch_is_about_to_create_is_still_approvable():
    """The check is on the PARENT directory, not the file: a patch may legitimately name a file it
    is about to create, and only a mangled path has a parent that does not exist either."""
    from orchestration import approvals
    assert approvals.unresolved_paths(["tests/test_not_written_yet.py"]) == []
    assert approvals.unresolved_paths(["ests/test_not_written_yet.py"]) == [
        "ests/test_not_written_yet.py"]


def test_every_approval_record_on_disk_round_trips():
    """No record may name a path that cannot be checked against the worktree. This is the test that
    would have caught AG-05 on the day it was written."""
    import json

    from orchestration import approvals
    recs = sorted(ROOT.glob("experiments/runs/*/approval_decision.json"))
    if not recs:
        import pytest
        pytest.skip("no approval records in this checkout")
    broken = {}
    for f in recs:
        paths = json.loads(f.read_text(encoding="utf-8")).get("paths", [])
        bad = approvals.unresolved_paths(paths)
        if bad:
            broken[f.parent.name] = bad
    assert not broken, f"approval records naming unresolvable paths: {broken}"
