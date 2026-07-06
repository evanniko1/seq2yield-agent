"""C7 demo CLI — carry a winning config from one scope onto another and test if it holds.

    # a config that won on cuperus_2017 -> does it beat sample_2019's own default?
    python scripts/run_config_transfer.py --model cnn --source cuperus_2017 --target sample_2019
    # transfer a specific series' winner onto another series
    python scripts/run_config_transfer.py --model rf --source ecoli --source-subregion 3 \
        --target ecoli --target-subregion 20
    # supply the config explicitly
    python scripts/run_config_transfer.py --model cnn --source yeast --target dream2022 \
        --config '{"kernel_sizes":[8,6,4],"lr":0.001}'

Source config comes from an explicit --config, else the winner of a recorded tournament on the
source, else a bounded C2 search. Reports whether it beats the target's default (C3 biology prior).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from seq2yield.experiments import config_transfer as C  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="C7 config_transfer")
    p.add_argument("--model", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--source-subregion", default=None)
    p.add_argument("--target-subregion", default=None)
    p.add_argument("--config", default=None, help="explicit JSON config to transfer")
    p.add_argument("--train-size", type=int, default=1000)
    p.add_argument("--feature-set", default="one_hot")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-record", action="store_true")
    args = p.parse_args()

    cfg = json.loads(args.config) if args.config else None
    r = C.transfer(args.model, source_dataset=args.source, target_dataset=args.target,
                   source_subregion=args.source_subregion, target_subregion=args.target_subregion,
                   config=cfg, train_size=args.train_size, feature_set=args.feature_set,
                   n_boot=args.n_boot, seed=args.seed, record=not args.no_record)

    src = args.source + (f":{args.source_subregion}" if args.source_subregion else "")
    tgt = args.target + (f":{args.target_subregion}" if args.target_subregion else "")
    print(f"\n=== config_transfer: {args.model}  {src} → {tgt} ===")
    print(f"source config ({r.source_of_config}): {json.dumps(r.source_config, default=str)}")
    print(f"target default:                       {json.dumps(r.target_default_config, default=str)}")
    print(f"\nR² transferred = {r.r2_transferred:.4f}   R² default = {r.r2_default:.4f}")
    print(f"ΔR² = {r.mean_delta:+.4f}  CI={r.ci}  (excludes 0: {r.excludes_zero}, n_test={r.n_test})")
    print(f"\nVERDICT: {r.verdict.upper().replace('_', ' ')}"
          + (f"  — the {src} winner transfers to {tgt}" if r.verdict == "beats_default"
             else f"  (min_delta={r.min_delta})"))


if __name__ == "__main__":
    main()
