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

from agents import council_eval as E  # noqa: E402


def main() -> None:
    res = E.run_ablation()
    print("\n=== council role ablation (fixed battery) ===")
    print(f"{'variant':<24}{'n_rev':>6}{'correct':>9}{'false_acc':>10}{'missed_val':>11}")
    for name, m in res.items():
        print(f"{name:<24}{m['n_reviewers']:>6}{m['correct_selection_rate']:>9}"
              f"{m['false_accept_rate']:>10}{m['mean_missed_value']:>11}")

    print("\n=== per-role contribution (drop the role → Δ) ===")
    for role, c in E.attribute_contributions(res).items():
        verdict = "REDUNDANT (covered by another role)" if c["false_accept_delta"] == 0 \
            else f"earns its cost (guards '{c['guards']}')"
        print(f"  {role:<22} guards={str(c['guards']):<12} "
              f"false_accept Δ={c['false_accept_delta']:+.2f}  → {verdict}")

    full, none = res["full"], res["no_critics"]
    print(f"\nfull panel: correct {full['correct_selection_rate']}, "
          f"false-accept {full['false_accept_rate']}   |   "
          f"no critics: correct {none['correct_selection_rate']}, "
          f"false-accept {none['false_accept_rate']}")
    print("(offline deterministic simulator; live mode runs the real Council under roles.configure)")


if __name__ == "__main__":
    main()
