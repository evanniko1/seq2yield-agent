"""R5 — per-role distilled memory (OpenOPC experience profiles): add/retrieve last-k lessons; the
reviewer prompt surfaces a role's lessons."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from agents import role_memory as RM  # noqa: E402


def test_add_and_retrieve_last_k(tmp_path):
    p = tmp_path / "rm.jsonl"
    for i in range(5):
        RM.add_lesson("methodology_reviewer", f"lesson {i}", path=p)
    RM.add_lesson("biology_reviewer", "bio lesson", path=p)
    ls = RM.lessons("methodology_reviewer", k=3, path=p)
    assert ls == ["lesson 2", "lesson 3", "lesson 4"]          # last 3, role-filtered
    assert RM.lessons("biology_reviewer", path=p) == ["bio lesson"]
    assert RM.lessons("doe_strategist", path=p) == []           # none yet


def test_as_block_empty_and_populated(tmp_path):
    p = tmp_path / "rm.jsonl"
    assert RM.as_block("chair", path=p) == ""
    RM.add_lesson("chair", "prefer the less confounded design on ties", path=p)
    b = RM.as_block("chair", path=p)
    assert "PRIOR LESSONS" in b and "less confounded" in b
