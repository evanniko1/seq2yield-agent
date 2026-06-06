"""Milestone 5: run the LLM council (generate -> review -> chair -> validated RunSpec).

Exit criterion: council proposes multiple experiments, rejects weak/confounded ones, and
emits one valid RunSpec.

Usage:
    python scripts/run_council.py                       # uses configured providers
    python scripts/run_council.py --allow-local-fallback --n 3   # offline (Ollama) demo
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8")           # avoid cp1252 errors on Windows
except Exception:
    pass

from agents.council import Council  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3, help="number of proposals to generate")
    ap.add_argument("--allow-local-fallback", action="store_true",
                    help="let authority roles borrow a local model when no API key is set "
                         "(DEV/offline only; marked in the audit trail)")
    args = ap.parse_args()

    stamp = f"{datetime.now(timezone.utc):%Y-%m-%d-%H%M%S}"
    out_dir = ROOT / "experiments/council_reviews" / stamp
    council = Council(allow_local_fallback=args.allow_local_fallback)
    res = council.run(n_proposals=args.n, out_dir=out_dir)

    print(f"generator: {res['generator']}   chair: {res.get('chair')}")
    nov = res.get("novelty", {})
    if nov:
        print(f"coverage: {nov.get('coverage')}")
        print(f"PI ({nov.get('pi')}) focus={nov.get('pi_focus')}: {str(nov.get('pi_rationale'))[:160]}")
        print(f"untested cells remaining: {nov.get('n_untested')}  dropped(settled/dupe): "
              f"{nov.get('dropped')}  kept_cells={nov.get('kept_cells')}")
    print(f"\n=== {res['n_proposals']} PROPOSALS ===")
    for p in res["proposals"]:
        s = res["mean_scores"].get(p["proposal_id"], {})
        print(f"  {p['proposal_id']}: {p['model_family']} vs {p['comparator_model']} "
              f"[{p['maturity_tier']}] - {p['title'][:60]}")
        print(f"     scores feas={s.get('feasibility')} value={s.get('scientific_value')} "
              f"clean={s.get('confoundedness')} repro={s.get('reproducibility')} "
              f"reject_votes={s.get('n_reject_votes')}")

    d = res["chair_decision"]
    print(f"\n=== CHAIR: {d['status'].upper()} (chose {d['chosen_proposal_id']}) ===")
    print(f"  {d['rationale'][:300]}")

    vr = res["runspec_validation"]
    print(f"\n=== RUNSPEC: {'VALID' if vr and vr.get('ok') else 'none/invalid'} ===")
    if res["runspec"]:
        rs = res["runspec"]
        print(f"  {rs['run_id']}: {rs['model_family']} vs "
              f"{rs['acceptance_policy']['baseline_model']} (baseline {rs['acceptance_policy']['baseline_run_id']})")
        if vr.get("warnings"):
            print(f"  warnings: {vr['warnings']}")
    print(f"\nartifacts: {out_dir}")
    print("Exit criterion (multiple proposals + reject weak + one valid RunSpec):",
          "MET" if (res["n_proposals"] >= 2 and vr and vr.get("ok")) else "partial")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
