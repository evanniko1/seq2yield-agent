"""Dissect a dataset's baselines into GENERATED questions (the exploratory arm of the harness).

    python scripts/run_insight.py --dataset ecoli
    python scripts/run_insight.py --dataset ecoli --metrics <a registry>/metrics.csv
        (default: the canonical registry in configs/baseline_registry.yaml; the
         dated 2026-06-04 registry this example used to name is now REFUSED as a
         control, see docs/RETRACTIONS.md R-5)

Reads the per-series baseline metrics a dataset already has and surfaces questions the council can
chase — turning reproduction/AutoML results into hypotheses. Writes experiments/insights/<dataset>.jsonl
and prints a ranked summary. Read-only over the metrics; owns no workflow state.

Since the 2026-11-02 insight rebuild (audit SA-6/ST-8/ML-10) each question names the DIFFICULTY
SCALE it was measured on, a covariate claim carries a family-wise max-T permutation p-value, and a
pooled dissection fits its neighbourhood partition on TRAIN sequences using two model families —
so `--families` is part of what a pooled run means, not a tuning knob.
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
    ap.add_argument("--families", default="rf,mlp",
                    help="model families for a pooled probe; >=2 is what lets a 'hard for every "
                         "model' question be emitted at all (audit ML-10)")
    args = ap.parse_args()
    families = tuple(f.strip() for f in args.families.split(",") if f.strip())

    metrics = Path(args.metrics) if args.metrics else dissect.default_metrics_path(args.dataset)
    if metrics and metrics.exists():                          # per-series: dissect baseline metrics
        questions = dissect.dissect_dataset(args.dataset, metrics, train_size=args.train_size)
        source = str(metrics.relative_to(ROOT))
    else:                                                     # pooled: discover neighborhoods first
        print(f"no per-series baseline for '{args.dataset}' — discovering neighborhoods from a "
              f"pooled baseline (fitting {args.k}-cluster probe on TRAIN sequences, "
              f"families={list(families)})...")
        questions = dissect.dissect_any(args.dataset, k=args.k, train_size=args.train_size,
                                        families=families)
        source = (f"discovered neighborhoods (pooled baseline, k={args.k}, "
                  f"families={','.join(families)})")
        if not questions:
            print(f"nothing to dissect for '{args.dataset}': no metrics.csv and no pooled data "
                  f"present. Reproduce baselines or check the dataset is onboarded.")
            return 1

    out = dissect.save_questions(args.dataset, questions)
    print(f"=== {len(questions)} generated question(s) for '{args.dataset}' (from {source}) ===\n")
    for q in questions:
        print(f"[{q.priority:.2f}] {q.kind}   (difficulty scale: {q.difficulty_scale})")
        print(f"   obs : {q.observation}")
        print(f"   why : {q.hypothesis}")
        print(f"   next: {q.suggested_intervention}"
              + (f"  (-> {q.intervention_type})" if q.intervention_type else ""))
        ev = q.evidence
        if ev.get("test") == "max_t_step_down_permutation":
            print(f"   test: max-T step-down permutation over {ev.get('n_candidates')} candidate(s) "
                  f"{ev.get('candidate_family')}, n={ev.get('n_units')} units, "
                  f"p={ev.get('p_max_t')}, reliability={ev.get('selection_reliability')}")
            if ev.get("dropped_covariates"):
                print(f"   drop: {sorted(ev['dropped_covariates'])}")
        if ev.get("test") == "y_matched_random_partition":
            print(f"   test: y-matched random partition, {ev.get('n_draw')} draws, "
                  f"harder-than-null units {ev.get('series')}")
        if ev.get("cluster_fit"):
            print(f"   fit : clusters fitted on {ev['cluster_fit']}"
                  + ("  [CAVEAT: " + ev["cluster_fit_caveat"] + "]"
                     if ev.get("cluster_fit_caveat") else ""))
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
