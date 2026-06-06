"""Show token + estimated-cost totals from the model-call log.

Usage: python scripts/show_cost.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orchestration import budget  # noqa: E402


def main() -> int:
    caps, prices = budget.load_config()
    records = budget.load_calls()
    s = budget.summarize(records, prices)
    print(f"MODEL-CALL COST  ({s['n_calls']} calls, {s['n_failed']} failed)")
    print(f"  tokens: {s['input_tokens']:,} in + {s['output_tokens']:,} out = {s['total_tokens']:,}")
    print(f"  estimated cost: ${s['total_cost_usd']:.4f}  "
          f"(caps: {caps['max_total_tokens']:,} tok / ${caps['max_total_cost_usd']} / {caps['max_calls']} calls)")
    for dim in ("by_provider", "by_role", "by_model"):
        print(f"\n  {dim.replace('by_', 'by ')}:")
        for name, d in sorted(s[dim].items(), key=lambda kv: -kv[1]["tokens"]):
            print(f"    {str(name):28s} calls={d['calls']:<4} tokens={d['tokens']:<9,} ${d['cost_usd']:.4f}")
    st = budget.BudgetTracker(caps, prices).status(records)
    print(f"\n  over_budget: {st['over_budget']}  {st['breaches'] or ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
