"""Canonical split registry.

Ingests the deposit's provided per-iteration splits (DECISIONS.md #9, #11) into a single
manifest under data/splits/. Each iteration is one Monte-Carlo CV repeat with a working set
(dev = train + hyperopt) and its own fixed held-out test set.

Once written, data/splits/ is strict-protected (configs/protected_files.yaml). Regenerating
from scratch is a verification path only; the provided splits are canonical.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from .cleaning import SERIES_COL


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def ingest_provided_splits(provided_root: str | Path, iterations, files_per_iteration: dict,
                           out_dir: str | Path) -> dict:
    provided_root = Path(provided_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    work_name = files_per_iteration["working_set"]
    held_name = files_per_iteration["heldout_set"]

    entries = {}
    hasher = hashlib.sha256()
    for it in iterations:
        wpath = provided_root / it / work_name
        hpath = provided_root / it / held_name
        if not wpath.exists() or not hpath.exists():
            raise FileNotFoundError(f"missing split files for {it}: {wpath} / {hpath}")
        w = pd.read_csv(wpath, usecols=[SERIES_COL])
        h = pd.read_csv(hpath, usecols=[SERIES_COL])
        w_sha, h_sha = _sha256(wpath), _sha256(hpath)
        hasher.update((w_sha + h_sha).encode())
        entries[it] = {
            "working_set": {"path": str(wpath), "sha256": w_sha, "n_rows": int(len(w))},
            "heldout_set": {"path": str(hpath), "sha256": h_sha, "n_rows": int(len(h))},
            "n_series": int(w[SERIES_COL].nunique()),
            "series": sorted(int(s) for s in w[SERIES_COL].unique()),
        }

    manifest = {
        "scheme": "per_series_provided",
        "n_iterations": len(iterations),
        "iterations": entries,
        "split_hash": hasher.hexdigest(),
    }
    (out_dir / "splits_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_manifest(out_dir: str | Path) -> dict:
    return json.loads((Path(out_dir) / "splits_manifest.json").read_text(encoding="utf-8"))
