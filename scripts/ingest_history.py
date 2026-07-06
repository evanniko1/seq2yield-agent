"""Build / refresh the local SQLite read-model from the existing artifact streams.

    python scripts/ingest_history.py            # (re)build reports/seq2yield.db from all history

Idempotent: re-run any time to fold in new runs. The dashboard (run_dashboard.py) reads this DB.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from orchestration import store  # noqa: E402


def main() -> None:
    con = store.connect()
    counts = store.ingest_all(con)
    print(f"ingested into {store.DB_PATH}:")
    for k, v in counts.items():
        print(f"  {k:<16} {v}")
    cost = store.cost_summary(con)
    print(f"\n{len(store.council_queries(con))} council queries · "
          f"{len(store.accepted_claims(con))} accepted claims · "
          f"{len(store.scoreboard(con))} tournaments · "
          f"${cost['total_cost_usd']} over {cost['n_calls']} model calls")
    con.close()


if __name__ == "__main__":
    main()
