"""Council evaluation + persona/role ablation (offline, deterministic).

    python scripts/run_council_eval.py

Runs the fixed battery under each role ablation and reports how much each critic role earns —
which flaws it uniquely guards, and where roles are redundant. Roles are data, so each ablation is
just a config diff (drop a reviewer / blank a persona).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse  # noqa: E402

from agents import council_eval as E  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="council role/persona ablation")
    ap.add_argument("--live", action="store_true",
                    help="run the REAL council (uses providers — costs API calls)")
    ap.add_argument("--structure", action="store_true",
                    help="provider-class ablations (authority-only / diversity-only) instead of per-role")
    args = ap.parse_args()

    variants = E.structure_variants() if args.structure else None
    if args.live:
        print("running LIVE ablation (real council, provider calls)…")
        res = E.run_live_ablation(variants=variants)
    else:
        res = E.run_ablation(variants=variants)
    label = "structure" if args.structure else "role"
    print(f"\n=== council {label} ablation ({'LIVE' if args.live else 'offline'}, fixed battery) ===")
    print(f"{'variant':<24}{'n_rev':>6}{'correct':>9}{'false_acc':>10}{'missed_val':>11}")
    for name, m in res.items():
        print(f"{name:<24}{m['n_reviewers']:>6}{m['correct_selection_rate']:>9}"
              f"{m['false_accept_rate']:>10}{m['mean_missed_value']:>11}")

    contrib = E.attribute_contributions(res)                  # empty unless per-role variants ran
    if contrib:
        print("\n=== per-role contribution (drop the role → Δ) ===")
        for role, c in contrib.items():
            verdict = "REDUNDANT (covered by another role)" if c["false_accept_delta"] == 0 \
                else f"earns its cost (guards '{c['guards']}')"
            print(f"  {role:<22} guards={str(c['guards']):<12} "
                  f"false_accept Δ={c['false_accept_delta']:+.2f}  → {verdict}")

    if "full" in res:
        f = res["full"]
        print(f"\nfull panel: correct {f['correct_selection_rate']}, "
              f"false-accept {f['false_accept_rate']}")
    print("(" + ("LIVE real council" if args.live else "offline deterministic simulator")
          + "; --live runs the real Council under roles.configure)")


if __name__ == "__main__":
    main()
