"""Tests for cost/token budget tracking (src/orchestration/budget.py)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orchestration import budget  # noqa: E402

PRICES = {"anthropic": {"input": 3.0, "output": 15.0},
          "openai": {"input": 2.5, "output": 10.0},
          "ollama": {"input": 0.0, "output": 0.0}}

RECS = [
    {"provider": "ollama", "model": "qwen", "role": "proposal_generator",
     "token_usage": {"input": 1000, "output": 500}, "success": True},
    {"provider": "anthropic", "model": "claude", "role": "chair",
     "token_usage": {"input": 1_000_000, "output": 100_000}, "success": True},
    {"provider": "openai", "model": "gpt", "role": "patch_reviewer",
     "token_usage": {"prompt_tokens": 200_000, "completion_tokens": 50_000}, "success": False},
]


def test_token_normalization_both_shapes():
    assert budget._tokens(RECS[1]) == (1_000_000, 100_000)      # input/output
    assert budget._tokens(RECS[2]) == (200_000, 50_000)         # prompt/completion


def test_cost_estimation():
    # anthropic: 1M*3 + 0.1M*15 = 3 + 1.5 = 4.5 ; openai: 0.2M*2.5 + 0.05M*10 = 0.5+0.5=1.0
    assert abs(budget.call_cost(RECS[1], PRICES) - 4.5) < 1e-9
    assert abs(budget.call_cost(RECS[2], PRICES) - 1.0) < 1e-9
    assert budget.call_cost(RECS[0], PRICES) == 0.0            # ollama free


def test_summarize_groups_and_totals():
    s = budget.summarize(RECS, PRICES)
    assert s["n_calls"] == 3 and s["n_failed"] == 1
    assert s["total_tokens"] == 1_000_000 + 100_000 + 1500 + 200_000 + 50_000
    assert abs(s["total_cost_usd"] - 5.5) < 1e-6
    assert set(s["by_provider"]) == {"ollama", "anthropic", "openai"}
    assert s["by_role"]["chair"]["cost_usd"] == 4.5


def test_budget_tracker_over_and_under():
    under = budget.BudgetTracker({"max_total_cost_usd": 100, "max_total_tokens": 10**9,
                                  "max_calls": 100}, PRICES).status(RECS)
    assert not under["over_budget"]
    over = budget.BudgetTracker({"max_total_cost_usd": 1.0, "max_total_tokens": 10**9,
                                 "max_calls": 100}, PRICES).status(RECS)
    assert over["over_budget"] and any("cost" in b for b in over["breaches"])
