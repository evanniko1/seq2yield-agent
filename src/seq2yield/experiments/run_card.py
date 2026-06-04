"""Run-card assembly (docs/CONTRACTS.md §5). The durable, machine-readable record of a run."""
from __future__ import annotations

import json
from pathlib import Path


def make_run_card(*, run_id: str, kind: str, dataset_hash: str, split_hash: str,
                  model_family: str, feature_set: str, train_sizes, seeds, iterations,
                  series, results: dict, environment: dict, status: str = "reproduction",
                  limitations=None) -> dict:
    return {
        "run_id": run_id,
        "kind": kind,
        "dataset_hash": dataset_hash,
        "split_hash": split_hash,
        "model_family": model_family,
        "feature_set": feature_set,
        "train_sizes": list(train_sizes),
        "seeds": list(seeds),
        "iterations": list(iterations),
        "n_series": len(series),
        "primary_metric": "r2",
        "results": results,
        "status": status,
        "limitations": limitations or [],
        "environment": environment,
    }


def write_run_card(card: dict, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "run_card.json"
    path.write_text(json.dumps(card, indent=2), encoding="utf-8")
    return path
