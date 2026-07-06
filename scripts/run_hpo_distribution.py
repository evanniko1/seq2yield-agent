"""C5 demo CLI — the per-series / per-subregion HPO-distribution study (the Nat Comms question).

    # distribution of the best CNN {kernel,lr,dropout} across E. coli series
    python scripts/run_hpo_distribution.py --dataset ecoli --model cnn --n-units 6
    # across a pooled dataset's GC strata
    python scripts/run_hpo_distribution.py --dataset sample_2019 --model rf --unit-type gc_bin

Searches each unit under the C10 gate (bounded/async) and reports which winning knobs vary across
units (heterogeneous) vs. share a sweet spot.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from seq2yield.experiments import hpo_distribution as H  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="C5 HPO-distribution study")
    p.add_argument("--dataset", required=True)
    p.add_argument("--model", default="cnn")
    p.add_argument("--unit-type", default="series", help="'series' or a stratum name (e.g. gc_bin)")
    p.add_argument("--n-units", type=int, default=8)
    p.add_argument("--train-size", type=int, default=800)
    p.add_argument("--min-action", default="light", choices=["light", "full"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-record", action="store_true")
    args = p.parse_args()

    res = H.run_hpo_distribution(args.dataset, args.model, unit_type=args.unit_type,
                                 n_units=args.n_units, train_size=args.train_size,
                                 min_action=args.min_action, seed=args.seed)

    print(f"\n=== HPO-distribution: {args.model} across {res.unit_type} on {args.dataset} "
          f"({len(res.per_unit)} units) ===")
    print(f"{'unit':<18}{'val R²':>9}  {'gate':<6} best (headline knobs)")
    for u in res.per_unit:
        knobs = {k: v for k, v in u.best_config.items()
                 if k in ("kernel_sizes", "lr", "dropout", "n_estimators", "max_depth", "d_model")}
        print(f"{u.unit:<18}{u.best_r2:>9.4f}  {u.gate_action:<6} {knobs}")

    print(f"\n--- distribution across units ({res.unit_type}) ---")
    for feat in (res.headline or list(res.distribution)):
        s = res.distribution.get(feat)
        if not s:
            continue
        het = "HETEROGENEOUS" if res.heterogeneous.get(feat) else "shared"
        if s["kind"] == "numeric":
            print(f"  {feat:<20} mean={s['mean']:<8} std={s['std']:<8} "
                  f"range=[{s['min']}, {s['max']}] cv={s['cv']:<6} -> {het}")
        else:
            print(f"  {feat:<20} mode={s['mode']:<10} counts={s['counts']} -> {het}")

    n_het = sum(1 for h in res.heterogeneous.values() if h)
    print(f"\n{n_het}/{len(res.heterogeneous)} knobs vary across {res.unit_type} "
          f"→ {'per-unit heterogeneity (no universal optimum)' if n_het else 'a shared optimum'}")
    if not args.no_record:
        print("saved:", H.record_study(res))


if __name__ == "__main__":
    main()
