"""Human-accept gate CLI for the expensive experiment modules.

    python scripts/run_queue.py --list                       # show the queue
    python scripts/run_queue.py --suggest tournament --params '{"dataset":"sample_2019"}' \
        --rationale "cnn vs flat on 5'UTR"                    # enqueue (does NOT run)
    python scripts/run_queue.py --accept 1a2b3c4d             # approve one
    python scripts/run_queue.py --reject 1a2b3c4d
    python scripts/run_queue.py --run                         # dispatch ALL accepted items

Only accepted items ever run; suggesting and accepting are logged to the RL-trace.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agents import experiment_queue as Q  # noqa: E402


def _print(recs):
    if not recs:
        print("(queue empty)")
        return
    for r in recs:
        res = f"  -> {json.dumps(r['result'])}" if r.get("result") else ""
        print(f"[{r['status']:>8}] {r['id']}  {r['kind']:<16} ~{r['estimated_cost_s']:.0f}s  "
              f"{json.dumps(r['params'])}\n           {r['rationale']}{res}")


def main() -> None:
    p = argparse.ArgumentParser(description="experiment human-accept gate")
    p.add_argument("--list", action="store_true")
    p.add_argument("--status", default=None, help="filter --list by status")
    p.add_argument("--suggest", default=None, choices=Q.KINDS)
    p.add_argument("--params", default="{}")
    p.add_argument("--rationale", default="manual suggestion")
    p.add_argument("--accept", default=None)
    p.add_argument("--reject", default=None)
    p.add_argument("--run", action="store_true", help="dispatch all accepted items")
    args = p.parse_args()

    if args.suggest:
        rec = Q.suggest(args.suggest, json.loads(args.params), args.rationale, source="cli")
        print(f"queued {rec['id']} ({rec['kind']}, ~{rec['estimated_cost_s']:.0f}s) — pending accept")
    if args.accept:
        r = Q.accept(args.accept)
        print(f"accepted {args.accept}" if r else f"no such id {args.accept}")
    if args.reject:
        r = Q.reject(args.reject)
        print(f"rejected {args.reject}" if r else f"no such id {args.reject}")
    if args.run:
        done = Q.run_accepted()
        print(f"dispatched {len(done)} accepted experiment(s):")
        _print(done)
    if args.list or not any((args.suggest, args.accept, args.reject, args.run)):
        _print(Q.list_queue(args.status))


if __name__ == "__main__":
    main()
