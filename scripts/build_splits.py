"""Milestone 2: ingest the deposit's provided splits into the canonical registry.

Writes data/splits/splits_manifest.json (per-iteration working/heldout paths, sha256, counts,
and a global split_hash). data/splits/ is strict-protected thereafter.

Usage: python scripts/build_splits.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seq2yield.data.splits import ingest_provided_splits  # noqa: E402


def main() -> int:
    cfg = (yaml.safe_load((ROOT / "configs/splits.yaml").read_text()) or {})["splits"]
    provided_root = ROOT / cfg["provided_root"]
    if not provided_root.exists():
        print(f"ERROR: {provided_root} missing. Run scripts/audit_archive.py first.", file=sys.stderr)
        return 2

    manifest = ingest_provided_splits(
        provided_root=provided_root,
        iterations=cfg["iterations"],
        files_per_iteration=cfg["files_per_iteration"],
        out_dir=ROOT / "data/splits",
    )
    print(f"[build_splits] {manifest['n_iterations']} iterations ingested")
    for it, e in manifest["iterations"].items():
        print(f"  {it}: working={e['working_set']['n_rows']} heldout={e['heldout_set']['n_rows']} "
              f"series={e['n_series']}")
    print(f"[build_splits] split_hash {manifest['split_hash'][:16]}...")
    print("[build_splits] wrote data/splits/splits_manifest.json (now strict-protected)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
