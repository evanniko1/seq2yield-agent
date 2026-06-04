"""Stage 0 archive audit CLI (Milestone 1).

Usage:
    python scripts/audit_archive.py
    python scripts/audit_archive.py --zip data/raw/seq2yield.zip --no-hash-members

Read-only over the archive. Emits the five manifests under data/manifests/, extracts data &
split CSVs to data/extracted/, and copies project notebooks (read-only) to
archive_notebooks_readonly/. See .claude/skills/audit-archive/SKILL.md.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seq2yield.data.manifests import run_audit  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 0 forensic archive audit.")
    ap.add_argument("--zip", default=str(ROOT / "data/raw/seq2yield.zip"))
    ap.add_argument("--manifests-dir", default=str(ROOT / "data/manifests"))
    ap.add_argument("--extract-dir", default=str(ROOT / "data/extracted"))
    ap.add_argument("--notebooks-dir", default=str(ROOT / "archive_notebooks_readonly"))
    ap.add_argument("--data-config", default=str(ROOT / "configs/data.yaml"))
    ap.add_argument("--no-hash-members", action="store_true",
                    help="Skip per-file sha256 (faster; inventory keeps crc32 only).")
    args = ap.parse_args()

    zip_path = Path(args.zip)
    if not zip_path.exists():
        print(f"ERROR: archive not found at {zip_path}. Fetch Zenodo DOI 10.5281/zenodo.7273952 "
              f"into data/raw/ first.", file=sys.stderr)
        return 2

    expected = {}
    cfg = Path(args.data_config)
    if cfg.exists():
        expected = (yaml.safe_load(cfg.read_text()) or {}).get("data", {})

    res = run_audit(
        zip_path=str(zip_path),
        manifests_dir=args.manifests_dir,
        extract_dir=args.extract_dir,
        notebooks_dir=args.notebooks_dir,
        expected=expected,
        hash_members=not args.no_hash_members,
    )

    print("\n=== AUDIT SUMMARY ===")
    print(f"archive sha256 : {res.archive_sha256}")
    print(f"real files     : {len(res.inventory)} ({res.n_junk} junk skipped)")
    print(f"datasets       : {[d['path'].split('/')[-1] for d in res.datasets]}")
    for d in res.datasets:
        print(f"  - {d['path'].split('/')[-1]}: {d['n_rows']} rows x {d['n_columns']} cols; "
              f"seq={d['sequence_columns']} target={d['target_columns']} series={d['series_columns']}")
    print(f"notebooks      : {len(res.notebooks)} project notebooks")
    print(f"split iters    : {sorted(res.splits)}")
    print(f"gaps logged    : {len(res.gaps)} (see data/manifests/reproducibility_gaps.md)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
