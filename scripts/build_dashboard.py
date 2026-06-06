"""M8: build the read-only research dashboard (static HTML) from the research trail.

Usage: python scripts/build_dashboard.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seq2yield.reporting.dashboard_export import build_dashboard  # noqa: E402


def main() -> int:
    out = build_dashboard()
    print(f"[dashboard] wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
