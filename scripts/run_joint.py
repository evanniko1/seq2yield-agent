"""C8 demo CLI — joint / cross-dataset training via length-reconciliation.

    # train on yeast 5'UTR, predict human 5'UTR (train-A -> test-B) in shared k-mer space
    python scripts/run_joint.py --train cuperus_2017 --test sample_2019
    # pool two sources, compare reconciliation strategies
    python scripts/run_joint.py --train cuperus_2017,yeast --test sample_2019 --compare
    python scripts/run_joint.py --train cuperus_2017 --test sample_2019 --strategy pad --model ridge
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from seq2yield.experiments import joint as J  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="C8 joint cross-dataset training")
    p.add_argument("--train", required=True, help="comma-separated source datasets")
    p.add_argument("--test", required=True, help="target dataset (train-A -> test-B)")
    p.add_argument("--model", default="rf")
    p.add_argument("--strategy", default="kmer", choices=list(J.STRATEGIES))
    p.add_argument("--train-size-per", type=int, default=2000)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--embed-model", default=None)
    p.add_argument("--compare", action="store_true", help="compare kmer vs adaptive_pool vs pad")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    train = args.train.split(",")
    print(f"\n=== joint: train {train} -> test {args.test}  ({args.model}) ===")
    if args.compare:
        for r in J.compare_strategies(train, args.test, model=args.model,
                                      strategies=("kmer", "adaptive_pool", "pad"),
                                      train_size_per=args.train_size_per, k=args.k, seed=args.seed):
            print(f"  {r.strategy:14s} dim={r.feature_dim:<6} Spearman={r.spearman}  R²_z={r.r2_z}"
                  + ("" if r.metric_primary == "spearman" else f"  [{r.metric_primary}]"))
    else:
        r = J.run_joint(train, args.test, model=args.model, strategy=args.strategy,
                        train_size_per=args.train_size_per, k=args.k, embed_model=args.embed_model,
                        seed=args.seed, record=True)
        print(f"strategy={r.strategy}  feature_dim={r.feature_dim}  "
              f"n_train={r.n_train}  n_test={r.n_test}")
        print(f"\nSpearman (cross-assay rank) = {r.spearman}    R² on z-target = {r.r2_z}")
        print("(targets z-scored per dataset before pooling; Spearman is the scale-free metric)")


if __name__ == "__main__":
    main()
