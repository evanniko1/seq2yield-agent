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
    args = ap.parse_args()

    metrics = Path(args.metrics) if args.metrics else dissect.default_metrics_path(args.dataset)
    if not metrics or not metrics.exists():
        print(f"no baseline metrics for '{args.dataset}'. Pass --metrics <run>/metrics.csv "
              f"(run scripts/reproduce_baselines.py first).")
        return 1

    questions = dissect.dissect_dataset(args.dataset, metrics, train_size=args.train_size)
    out_dir = ROOT / "experiments/insights"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.dataset}.jsonl"
    out.write_text("\n".join(q.model_dump_json() for q in questions), encoding="utf-8")

    print(f"=== {len(questions)} generated question(s) for '{args.dataset}' "
          f"(from {metrics.relative_to(ROOT)}) ===\n")
    for q in questions:
        print(f"[{q.priority:.2f}] {q.kind}")
        print(f"   obs : {q.observation}")
        print(f"   why : {q.hypothesis}")
        print(f"   next: {q.suggested_intervention}"
              + (f"  (-> {q.intervention_type})" if q.intervention_type else ""))
        print()
    hints = dissect.to_focus_hints(questions)
    print(f"data-driven focus hints for the PI: {hints or '(none)'}")
    print(f"written: {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
