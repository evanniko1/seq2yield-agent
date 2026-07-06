"""Reconstruct one council query's full trail from the read-model — the terminal-native counterpart
to the dashboard's query page (adopted from shepherd's `run show`; see docs/DECISIONS.md).

    python scripts/run_show.py                     # list council queries
    python scripts/run_show.py <trajectory_id>     # full trail: proposals, reviews, decision trace
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from orchestration import store  # noqa: E402


def main() -> None:
    con = store.connect()
    store.ingest_all(con)
    if len(sys.argv) < 2:
        print("council queries (pass an id for the full trail):")
        for q in store.council_queries(con):
            print(f"  {q['trajectory_id']:<28} {q['ts'] or '':<20} chair={q.get('chair_status') or '—'}")
        return
    d = store.query_detail(con, sys.argv[1])
    if not d["query"]:
        print("no such trajectory:", sys.argv[1])
        return
    q = d["query"]
    print(f"\n=== {sys.argv[1]} ===")
    print(f"chair: {q.get('chair_status') or '—'}   chosen: {q.get('chosen_proposal_id') or '—'}")
    print("\nproposals:")
    for p in d["proposals"]:
        print(f"  [{p['proposal_id']}] {p['model_family']} vs {p['comparator_model']} "
              f"({p['intervention_type']}, {p['dataset']}): {p['hypothesis']}")
    print("\nreviews:")
    for r in d["reviews"]:
        print(f"  {r['proposal_id']}  {r['role']:<22} feas={r['feasibility']} val={r['scientific_value']} "
              f"clean={r['confoundedness']} repro={r['reproducibility']}"
              + (f"  REJECT: {r['reject_reason']}" if r['reject_reason'] else ""))
    print("\ndecision trail (RL-trace):")
    for e in d["events"]:
        print(f"  {e['ts'] or '':<20} {e['decision_type']:<22} → {e['selected_action']}  ({e['policy']})")


if __name__ == "__main__":
    main()
