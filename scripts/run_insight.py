"""Dissect a dataset's baselines into GENERATED questions (the exploratory arm of the harness).

    python scripts/run_insight.py --dataset ecoli
    python scripts/run_insight.py --dataset ecoli --metrics experiments/runs/2026-06-04-full56/metrics.csv

Reads the per-series baseline metrics a dataset already has and surfaces questions the council can
chase — turning reproduction/AutoML results into hypotheses. Writes experiments/insights/<dataset>.jsonl
and prints a ranked summary. Read-only over the metrics; owns no workflow state.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:                                    # the questions carry ΔR²/² etc.; a cp1252 console can't print them
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from seq2yield.insight import dissect  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ecoli")
    ap.add_argument("--metrics", help="path to the baseline run's metrics.csv (per-series R²)")
    ap.add_argument("--train-size", type=int, default=None)
    ap.add_argument("--k", type=int, default=8, help="neighborhoods to discover (pooled datasets)")
    args = ap.parse_args()

    metrics = Path(args.metrics) if args.metrics else dissect.default_metrics_path(args.dataset)
    if metrics and metrics.exists():                          # per-series: dissect baseline metrics
        questions = dissect.dissect_dataset(args.dataset, metrics, train_size=args.train_size)
        source = str(metrics.relative_to(ROOT))
    else:                                                     # pooled: discover neighborhoods first
        print(f"no per-series baseline for '{args.dataset}' — discovering neighborhoods from a "
              f"pooled baseline (fitting {args.k}-cluster probe)...")
        questions = dissect.dissect_any(args.dataset, k=args.k, train_size=args.train_size)
        source = f"discovered neighborhoods (pooled baseline, k={args.k})"
        if not questions:
            print(f"nothing to dissect for '{args.dataset}': no metrics.csv and no pooled data "
                  f"present. Reproduce baselines or check the dataset is onboarded.")
            return 1

    out = dissect.save_questions(args.dataset, questions)
    print(f"=== {len(questions)} generated question(s) for '{args.dataset}' (from {source}) ===\n")
    for q in questions:
        print(f"[{q.priority:.2f}] {q.kind}")
        print(f"   obs : {q.observation}")
        print(f"   why : {q.hypothesis}")
        print(f"   next: {q.suggested_intervention}"
              + (f"  (-> {q.intervention_type})" if q.intervention_type else ""))
        print()
    hints = dissect.to_focus_hints(questions)
    print(f"data-driven focus hints for the PI: {hints or '(none)'}")
    try:                                                     # two-phase exploration state
        from agents import memory
        from seq2yield.insight import dataset_phase
        ph = dataset_phase(args.dataset, memory.load())
        print(f"exploration phase: {ph['phase']}  ({ph['reason']})")
        if ph["neighborhoods"]:
            print(f"  focus neighborhoods: {ph['neighborhoods'][:12]}")
    except Exception:
        pass
    print(f"written: {out.relative_to(ROOT)}  (the PI reads this next cycle)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
