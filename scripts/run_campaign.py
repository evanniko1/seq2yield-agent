"""Autonomous research campaign: run agentic cycles until a stopping rule fires.

The council decides WHAT to run (coverage-driven, revisiting inconclusive cells with more
power); the campaign decides WHEN TO STOP, instead of a human picking a fixed cycle count.

Stopping rules (any triggers):
  - coverage target reached (settled fraction >= --coverage-target)
  - all cells resolved (no untested and no inconclusive)
  - diminishing returns (no new settled cell for --patience cycles)
  - safety cap (--max-cycles)

Usage:
    python scripts/run_campaign.py --allow-local-fallback --max-cycles 8 --coverage-target 0.5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import run_agent_loop as loop  # noqa: E402
from agents import memory, question_space  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-local-fallback", action="store_true")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--max-cycles", type=int, default=8)
    ap.add_argument("--coverage-target", type=float, default=0.5, help="settled fraction to stop")
    ap.add_argument("--patience", type=int, default=3, help="stop after N cycles with no new settled cell")
    args = ap.parse_args()

    no_progress = 0
    for c in range(1, args.max_cycles + 1):
        summ = question_space.summarize(memory.load())
        frac = summ["settled"] / max(1, summ["total_cells"])
        print(f"\n############ CAMPAIGN CYCLE {c}/{args.max_cycles} "
              f"(settled {summ['settled']}/{summ['total_cells']} = {summ['coverage_pct']}%, "
              f"inconclusive {summ['inconclusive']}, untested {summ['untested']}) ############")

        if frac >= args.coverage_target:
            print(f"STOP: coverage target {args.coverage_target:.0%} reached."); break
        if summ["untested"] == 0 and summ["inconclusive"] == 0:
            print("STOP: all catalogue cells resolved (settled)."); break

        before = summ["settled"]
        loop.cycle(args.allow_local_fallback, n_proposals=args.n)
        after = question_space.summarize(memory.load())["settled"]
        no_progress = no_progress + 1 if after <= before else 0
        if no_progress >= args.patience:
            print(f"STOP: diminishing returns ({args.patience} cycles, no new settled cell)."); break
    else:
        print("STOP: reached max-cycles safety cap.")

    final = question_space.summarize(memory.load())
    print(f"\n=== CAMPAIGN COMPLETE: {final['settled']}/{final['total_cells']} settled "
          f"({final['coverage_pct']}%), {final['inconclusive']} inconclusive, "
          f"{final['untested']} untested ===")
    print("run `python scripts/show_coverage.py` for the full frontier.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
