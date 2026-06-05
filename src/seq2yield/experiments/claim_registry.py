"""Claim registry (docs/PROJECT_SPEC.md §; AGENTS.md §0).

The durable ledger of scientific claims the workflow is willing to make. A claim is only
recorded when the harness ACCEPTED the run — no claim without run-card evidence. Rejected /
inconclusive runs are logged too (with claim=null) so the trail is complete.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CLAIMS_DIR = ROOT / "experiments" / "claims"
REGISTRY = CLAIMS_DIR / "registry.jsonl"


def record(*, run_id: str, proposal_id: str, status: str, comparison: dict,
           claim_allowed: str | None, claims_dir: str | Path | None = None) -> dict:
    claims_dir = Path(claims_dir or CLAIMS_DIR)
    claims_dir.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "proposal_id": proposal_id,
        "status": status,
        "candidate_model": comparison.get("candidate_model"),
        "baseline_model": comparison.get("baseline_model"),
        "mean_delta_r2": comparison.get("mean_delta"),
        "bootstrap_ci_95": comparison.get("paired_bootstrap_ci"),
        "ci_excludes_zero": comparison.get("ci_excludes_zero"),
        "n_series": comparison.get("n_series"),
        "train_size": comparison.get("comparison_train_size"),
        # claim only survives if the harness accepted the run
        "claim": claim_allowed if status == "accepted" else None,
    }
    (claims_dir / f"{run_id}.json").write_text(json.dumps(entry, indent=2), encoding="utf-8")
    with open(claims_dir / "registry.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def load(claims_dir: str | Path | None = None) -> list[dict]:
    reg = Path(claims_dir or CLAIMS_DIR) / "registry.jsonl"
    if not reg.exists():
        return []
    return [json.loads(x) for x in reg.read_text(encoding="utf-8").splitlines() if x.strip()]


def accepted_claims(claims_dir: str | Path | None = None) -> list[dict]:
    return [c for c in load(claims_dir) if c.get("claim")]
