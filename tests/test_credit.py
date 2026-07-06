"""R3 — online per-role credit assignment: alignment aggregation + building votes from real reviews
vs realized outcomes (a reviewer is right when it vouched-for an accepted run or flagged a rejected
one). The online complement to the offline persona ablation."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agents import credit  # noqa: E402


def test_assign_credit_aggregates_alignment():
    cycles = [
        {"role_votes": [{"role": "A", "aligned": True}, {"role": "B", "aligned": False}]},
        {"role_votes": [{"role": "A", "aligned": True}, {"role": "B", "aligned": False}]},
    ]
    c = credit.assign_credit(cycles)
    assert c["A"]["credit"] == 1.0 and c["B"]["credit"] == 0.0
    assert c["A"]["n"] == 2


def test_credit_from_reviews_rewards_aligned_judgments():
    cycles = [
        # accepted run: 'aligned' vouched-for it (right); 'contrarian' rejected it (wrong)
        {"outcome": "accepted", "reviews": [
            {"role": "aligned", "score_confoundedness": 5, "score_feasibility": 5},
            {"role": "contrarian", "score_confoundedness": 5, "score_feasibility": 5, "reject_reason": "nah"}]},
        # rejected run: 'aligned' flagged it (right); 'contrarian' vouched-for it (wrong)
        {"outcome": "rejected", "reviews": [
            {"role": "aligned", "score_confoundedness": 1, "score_feasibility": 5, "reject_reason": "confound"},
            {"role": "contrarian", "score_confoundedness": 5, "score_feasibility": 5}]},
        {"outcome": "inconclusive", "reviews": [{"role": "aligned"}]},   # skipped (no ground truth)
    ]
    c = credit.credit_from_reviews(cycles)
    assert c["aligned"]["credit"] == 1.0      # vouched the good, flagged the bad
    assert c["contrarian"]["credit"] == 0.0   # backwards both times
    assert c["aligned"]["n"] == 2             # the inconclusive cycle contributed nothing


def test_empty_is_empty():
    assert credit.assign_credit([]) == {} and credit.credit_from_reviews([]) == {}
