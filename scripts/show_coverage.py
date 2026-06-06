"""Show the council's question-space coverage from research memory.

Usage: python scripts/show_coverage.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from agents import memory, question_space  # noqa: E402


def main() -> int:
    recs = memory.load()
    summ = question_space.summarize(recs)
    cov = question_space.coverage(recs)
    print(f"QUESTION-SPACE COVERAGE  ({summ['settled']}/{summ['total_cells']} settled = "
          f"{summ['coverage_pct']}%, {summ['inconclusive']} inconclusive, "
          f"{summ['untested']} untested)\n")
    for status in ("settled", "inconclusive", "untested"):
        ids = [cid for cid, e in cov.items() if e["status"] == status]
        print(f"== {status.upper()} ({len(ids)}) ==")
        for cid in sorted(ids):
            e = cov[cid]
            extra = ""
            if e["n_runs"]:
                extra = f"  runs={e['n_runs']} last_Δ={e['last_delta']} {e['statuses']}"
            print(f"  {cid}  — {e['describe']}{extra}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
