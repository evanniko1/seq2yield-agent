"""C4 demo CLI — the best-algorithm-per-scope tournament.

    python scripts/run_tournament.py --dataset sample_2019
    python scripts/run_tournament.py --dataset ecoli --n-series 6          # series-unit
    python scripts/run_tournament.py --dataset ecoli --subregion 3         # single series
    python scripts/run_tournament.py --dataset cuperus_2017 --family cnn,rf,mlp,ridge --tune

Ranks the family by held-out R², paired-bootstraps the winner vs each rival, BH-FDR-corrects the
family, and records the leaderboard + headline winner claim.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from seq2yield.experiments import tournament as T  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="C4 best-algorithm-per-scope tournament")
    p.add_argument("--dataset", required=True)
    p.add_argument("--subregion", default=None)
    p.add_argument("--family", default=None, help="comma-separated model list (default ridge,rf,mlp,cnn)")
    p.add_argument("--feature-set", default="one_hot")
    p.add_argument("--train-size", type=int, default=1000)
    p.add_argument("--n-series", type=int, default=8)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--tune", action="store_true", help="tune each contender via C3→C10→C2")
    p.add_argument("--control", action="store_true",
                   help="run the shuffled-label negative control on the winner (leakage sanity)")
    p.add_argument("--no-record", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    family = args.family.split(",") if args.family else None
    res = T.run_tournament(args.dataset, subregion=args.subregion, family=family,
                           feature_set=args.feature_set, train_size=args.train_size,
                           n_series=args.n_series, n_boot=args.n_boot, tune=args.tune, seed=args.seed)
    scope = f"{args.dataset}" + (f"/series={args.subregion}" if args.subregion is not None else "")
    print(f"\n=== tournament: {scope}  [{res.scope}, {res.bootstrap_unit}-unit, "
          f"n={res.n_units}, train={res.train_size}, selection={res.selection}] ===")
    valcol = "R²_val" if res.selection == "nested_val" else ""
    print(f"{'rank':>4}  {'model':<12}{'R²_test':>9}{valcol:>9}{'Δ_vs_win':>10}{'q_value':>9}  fdr  source")
    for c in res.leaderboard:
        q = "" if c.q_value is None else f"{c.q_value:.3f}"
        fdr = "" if c.survives_fdr is None else ("yes" if c.survives_fdr else "no ")
        params = f"  [{c.n_params:,}p]" if c.n_params else ""
        v = "" if c.r2_val is None else f"{c.r2_val:.4f}"
        print(f"{c.rank:>4}  {c.model:<12}{c.r2:>9.4f}{v:>9}{(c.delta_vs_winner or 0):>10.4f}"
              f"{q:>9}  {fdr:>3}  {c.hyperparameters_source}{params}")
    print(f"\nWINNER: {res.winner}"
          + (f"  — SIGNIFICANT over {res.runner_up} (Δ≥{res.min_delta}, survives BH-FDR)"
             if res.winner_significant else
             f"  — leads {res.runner_up} but NOT significant after correction"))
    if res.selection == "nested_val":
        print("(ranked on a validation slice; test R² reported for the already-chosen winner — no selection-on-test)")
    if args.control:
        from seq2yield.experiments import controls as C
        r2c = C.shuffled_label_r2(args.dataset, res.winner, subregion=args.subregion,
                                  train_size=args.train_size, feature_set=args.feature_set, seed=args.seed)
        print(f"negative control (shuffled labels, {res.winner}): R²={r2c:+.4f} "
              f"→ {'OK (no leakage)' if C.negative_control_ok(r2c) else 'WARNING: leakage/bug'}")
    if not args.no_record:
        T.record_tournament(res)
        print("recorded to the claim ledger (+ experiments/claims/tournaments.jsonl)")


if __name__ == "__main__":
    main()
