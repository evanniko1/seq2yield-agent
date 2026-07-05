"""C2 demo CLI — run a hybrid hyperparameter search and print the argmax config + trace.

    python scripts/run_search.py --model cnn --dataset sample_2019 --strategy bandit --n-trials 12
    python scripts/run_search.py --model rf  --dataset cuperus_2017 --seeds '{"n_estimators":400}'

Scoring uses a validation split of the training data only (never the test set). The winning config
is what C10/the council would hand to the harness for a full, powered run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from seq2yield.search import SearchBudget, search  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="C2 hybrid hyperparameter search")
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--subregion", default=None)
    p.add_argument("--strategy", default="random", choices=["random", "bandit"])
    p.add_argument("--feature-set", default="one_hot")
    p.add_argument("--feature-scaling", default="none")
    p.add_argument("--n-trials", type=int, default=16)
    p.add_argument("--max-train-size", type=int, default=4000)
    p.add_argument("--score-epochs", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seeds", default=None,
                   help="JSON config or list of configs to warm-start (LLM-guided)")
    args = p.parse_args()

    seeds = None
    if args.seeds:
        parsed = json.loads(args.seeds)
        seeds = parsed if isinstance(parsed, list) else [parsed]

    budget = SearchBudget(n_trials=args.n_trials, max_train_size=args.max_train_size,
                          score_epochs=args.score_epochs)
    r = search(args.model, args.dataset, subregion=args.subregion, strategy=args.strategy,
               feature_set=args.feature_set, feature_scaling=args.feature_scaling,
               seed=args.seed, seeds=seeds, budget=budget)

    print(f"\n=== {args.model} / {args.dataset}"
          f"{'/' + args.subregion if args.subregion else ''} "
          f"[{args.strategy}] ===")
    print(f"best validation R² = {r.best_score:.4f}  ({r.n_evals} evals, "
          f"{r.seeds_used} LLM seed(s))")
    print("best config:")
    print(json.dumps(r.best_config, indent=2, default=str))
    print("\ntrace (phase: R²):")
    for t in r.trace:
        print(f"  {t['phase']:>8}  n={t['train_size']:<6}  R²={t['score']:.4f}")


if __name__ == "__main__":
    main()
