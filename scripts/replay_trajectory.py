"""Replay a council trajectory from the decision-event trace (RL-readiness).

The practical RL-readiness test: "can you replay a council run and explain why each agent/model/
action was chosen?" This prints the ordered decisions for a trajectory and, with --rows, the
RL-ready extraction (state_features, action_taken, candidate_actions, outcome_metrics,
reward_proxy). NO training — pure traceability.

Usage:
  python scripts/replay_trajectory.py --list
  python scripts/replay_trajectory.py <trajectory_id>
  python scripts/replay_trajectory.py <trajectory_id> --rows
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from agents import trace  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("trajectory_id", nargs="?")
    ap.add_argument("--list", action="store_true", help="list known trajectories")
    ap.add_argument("--rows", action="store_true", help="print RL-ready training rows")
    args = ap.parse_args()

    events = trace.read_events()
    if args.list or not args.trajectory_id:
        seen = {}
        for e in events:
            seen.setdefault(e.get("run_id"), 0)
            seen[e["run_id"]] += 1
        print(f"{len(seen)} trajectories in {trace.EVENTS_PATH}:")
        for tid, n in seen.items():
            print(f"  {tid}: {n} decisions")
        return 0

    print(trace.replay(args.trajectory_id))
    if args.rows:
        from agents import memory
        rows = [r for r in trace.extract_training_rows(memory_records=memory.load())
                if r["trajectory_id"] == args.trajectory_id]
        print("\nRL-ready rows (state_features, action_taken, candidate_actions, outcome, reward_proxy):")
        print(json.dumps(rows, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
