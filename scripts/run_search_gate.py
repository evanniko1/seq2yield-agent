"""C10 demo CLI — the Council's search-worthiness decision (+ optional bounded/async search).

    # decision only (no training): should SKIP a barely-tunable / decisive cell, run a hot one
    python scripts/run_search_gate.py --model cnn --dataset sample_2019 --inconclusive --overfit
    python scripts/run_search_gate.py --model ridge --dataset sample_2019          # -> skip
    # actually execute the gated search, bounded + async (never hangs):
    python scripts/run_search_gate.py --model rf --dataset sample_2019 --inconclusive --execute

The decision is logged to reports/decision_events.jsonl (RL-trace); on skip/timeout the run would
use C1 defaults.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agents import memory, search_gate  # noqa: E402
from agents.council import _min_delta_r2  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="C10 search-worthiness gate")
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--subregion", default=None)
    p.add_argument("--intervention", default="training_procedure")
    p.add_argument("--inconclusive", action="store_true", help="force the inconclusive signal")
    p.add_argument("--overfit", action="store_true")
    p.add_argument("--data-limited", action="store_true")
    p.add_argument("--execute", action="store_true", help="actually run the gated search (bounded)")
    args = p.parse_args()

    ctx = search_gate.build_context(args.model, args.dataset, subregion=args.subregion,
                                    intervention_type=args.intervention,
                                    min_delta=_min_delta_r2(), memory_records=memory.load(),
                                    overfit=args.overfit, data_limited=args.data_limited)
    if args.inconclusive:
        ctx.inconclusive = True

    decision = search_gate.decide(ctx)
    print(f"\n=== C10 gate: {args.model} / {args.dataset} [{args.intervention}] ===")
    print(f"value-of-information = {decision.value_score:.2f}   cost/budget = {decision.cost_score:.2f}")
    print(f"DECISION: {decision.action.upper()}  — {decision.reason}")
    if decision.budget:
        print(f"budget: n_trials={decision.budget.n_trials}, "
              f"max_train_size={decision.budget.max_train_size}, deadline≈{decision.deadline_s:.0f}s")

    if args.execute:
        print("\nrunning gated search (bounded + async)…")
        out = search_gate.run_gated(ctx)
        if out.decision.action == "skip":
            print("skipped — would use C1 defaults")
        elif out.timed_out:
            print("timed out within deadline — loop proceeds with C1 defaults (never hangs)")
        else:
            print(f"best validation R² = {out.result.best_score:.4f} "
                  f"({out.result.n_evals} evals)")
            print("winning config:", json.dumps(out.result.best_config, indent=2, default=str))
    else:
        # decision-only: log it (RL-trace) without training
        search_gate.trace.log_event(
            "search_worthiness", candidate_actions=["skip", "light", "full"],
            selected_action=decision.action, policy="c10_gate_v1", reason=decision.reason,
            state={"model": args.model, "dataset": args.dataset,
                   "value_score": round(decision.value_score, 3)})
        print("(decision logged to reports/decision_events.jsonl; pass --execute to run the search)")


if __name__ == "__main__":
    main()
