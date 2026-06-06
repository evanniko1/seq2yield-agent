"""Show the claim registry with multiple-comparison (BH-FDR) correction (CRITIQUE C1).

Each council comparison decided significance individually at α=0.05; across the whole family
that inflates false positives. This re-evaluates the family with Benjamini-Hochberg FDR (or
Bonferroni) and shows which raw discoveries survive correction.

Usage:
    python scripts/show_claims.py [--method bh|bonferroni] [--alpha 0.05]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from agents import memory  # noqa: E402
from seq2yield.experiments import claim_registry  # noqa: E402
from seq2yield.statistics import multiple_comparisons as mc  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["bh", "bonferroni"], default="bh")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--source", choices=["claims", "memory"], default="memory")
    args = ap.parse_args()

    records = memory.load() if args.source == "memory" else claim_registry.load()
    res = mc.correct_claims(records, alpha=args.alpha, method=args.method)

    print(f"FAMILY-WISE CORRECTION ({res['method']}, alpha={res['alpha']}) over "
          f"{res['n_comparisons']} comparisons with p-values")
    print(f"  raw discoveries (individually significant): {res['n_raw_discoveries']}")
    print(f"  survive correction:                         {res['n_after_correction']}  "
          f"(threshold p<= {res['threshold']:.4g})")
    if res["runs_without_pvalue"]:
        print(f"  {len(res['runs_without_pvalue'])} run(s) have no p-value (pre-C1) — excluded")
    print("\n  run_id                                        Δ        p       q      raw  survives")
    for a in sorted(res["runs"], key=lambda x: (x["p_value"] if x["p_value"] is not None else 1)):
        d = a["mean_delta"]
        ds = f"{d:+.3f}" if isinstance(d, (int, float)) else "  —  "
        print(f"  {str(a['run_id'])[:44]:44s} {ds}  {a['p_value']:.3g}  {a['q_value']:.3g}  "
              f"{'Y' if a['raw_discovery'] else '·'}    {'Y' if a['survives_correction'] else '·'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
