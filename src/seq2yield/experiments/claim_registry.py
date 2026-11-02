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


def _provenance() -> dict:
    """The code identity behind the claim: git SHA plus the fingerprint of the value-deciding
    files. Cheap (cached) and never allowed to break a recording."""
    try:
        from ..training import provenance
        return {"git_sha": provenance.code_version().get("git_sha_short"),
                "numerics": provenance.numerics_fingerprint()["fingerprint"]}
    except Exception:
        return {}


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
        "p_value": comparison.get("p_value"),
        "bootstrap_unit": comparison.get("bootstrap_unit", "series"),   # C3: comparability fence
        # The PROTOCOL the number was measured under, so a claim can never again be quoted without
        # it (docs/RETRACTIONS.md). A record with split_strategy=None predates 2026-11-01.
        "split_strategy": (comparison.get("split") or {}).get("split_strategy"),
        "n_bootstrap_units": comparison.get("n_bootstrap_units"),
        # ...and the ESTIMATOR, for the same reason: a t-interval + exact sign-flip verdict is not
        # comparable with a percentile one, and the percentile method was liberal at these unit
        # counts (audit ST-2). ci_method=None likewise predates 2026-11-02.
        "ci_method": comparison.get("ci_method"),
        "p_method": comparison.get("p_method"),
        "p_floor": comparison.get("p_floor"),
        # ...and the verdict ABOUT THE MARGIN plus the margin itself (audit ST-3). `accepted` used
        # to mean "point estimate over the bar", which a true delta under the bar passed ~42% of
        # the time; it now means the one-sided lower limit cleared it. margin_verdict=None
        # identifies a claim decided by the old rule.
        "margin_verdict": comparison.get("margin_verdict"),
        "margin": comparison.get("margin"),
        "ci_one_sided": comparison.get("ci_one_sided"),
        # WHAT it was compared against, and WHICH code produced it. Eight claims were recorded
        # against a registry that had drifted out from under them with no way to tell (ML-02).
        "baseline_source": comparison.get("baseline_source"),
        "provenance": _provenance(),
        "dataset": comparison.get("dataset", "ecoli"),                  # K1: organism dimension
        "transfer": (comparison.get("transfer") or {}).get("verdict"),  # K1: replication verdict
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
