"""Multiple-comparison correction over the family of council comparisons (CRITIQUE C1).

The council runs many comparisons across the question-space grid and each accepts/rejects at
α=0.05 individually, so the family-wise false-positive rate is uncontrolled. This applies
Benjamini-Hochberg FDR (default) or Bonferroni over the p-values of all directional runs and
reports which "discoveries" survive correction.
"""
from __future__ import annotations


def benjamini_hochberg(pvalues: list[float], alpha: float = 0.05) -> dict:
    """BH-FDR. Returns rejected flags (original order), BH-adjusted q-values, and the threshold."""
    m = len(pvalues)
    if m == 0:
        return {"rejected": [], "qvalues": [], "threshold": 0.0, "n": 0, "n_rejected": 0}
    order = sorted(range(m), key=lambda i: pvalues[i])
    # step-up: largest k with p_(k) <= (k/m)*alpha
    crit_idx = -1
    for rank, i in enumerate(order, start=1):
        if pvalues[i] <= rank / m * alpha:
            crit_idx = rank
    threshold = (crit_idx / m * alpha) if crit_idx > 0 else 0.0
    rejected = [pvalues[i] <= threshold for i in range(m)]
    # monotone BH-adjusted q-values
    q = [0.0] * m
    prev = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        prev = min(prev, pvalues[i] * m / rank)
        q[i] = min(1.0, prev)
    return {"rejected": rejected, "qvalues": q, "threshold": threshold,
            "n": m, "n_rejected": sum(rejected), "alpha": alpha, "method": "benjamini_hochberg"}


def bonferroni(pvalues: list[float], alpha: float = 0.05) -> dict:
    m = len(pvalues)
    thr = alpha / m if m else 0.0
    rejected = [p <= thr for p in pvalues]
    return {"rejected": rejected, "qvalues": [min(1.0, p * m) for p in pvalues],
            "threshold": thr, "n": m, "n_rejected": sum(rejected), "alpha": alpha,
            "method": "bonferroni"}


def gather_family(claims_dir=None) -> list[dict]:
    """The FULL family of directional tests to correct JOINTLY (G2): the claim registry AND the
    tournament winner-vs-runner-up tests (previously each corrected in isolation → under-corrected).
    Each item is {id, source, p_value, status}."""
    import json
    from pathlib import Path

    from ..experiments import claim_registry
    cd = Path(claims_dir or claim_registry.CLAIMS_DIR)
    fam = []
    for c in claim_registry.load(cd):
        if isinstance(c.get("p_value"), (int, float)):
            fam.append({"id": c.get("run_id"), "source": "claim",
                        "p_value": float(c["p_value"]), "status": c.get("status")})
    tf = cd / "tournaments.jsonl"
    if tf.exists():
        for line in tf.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            t = json.loads(line)
            ru = next((x for x in (t.get("leaderboard") or []) if x.get("rank") == 2), None)
            if ru and isinstance(ru.get("p_value"), (int, float)):
                fam.append({"id": t.get("run_id"), "source": "tournament",
                            "p_value": float(ru["p_value"]),
                            "status": "accepted" if t.get("winner_significant") else "inconclusive"})
    return fam


def correct_all(claims_dir=None, *, alpha: float = 0.05, method: str = "bh") -> dict:
    """BH-FDR (or Bonferroni) over the JOINT family of claims + tournament headline tests (G2)."""
    fam = gather_family(claims_dir)
    pvals = [f["p_value"] for f in fam]
    res = (bonferroni if method == "bonferroni" else benjamini_hochberg)(pvals, alpha)
    items = []
    for f, q, rej in zip(fam, res["qvalues"], res["rejected"]):
        items.append({**f, "q_value": round(q, 4), "survives_correction": bool(rej),
                      "raw_discovery": f["status"] in ("accepted", "rejected")})
    return {"method": res.get("method"), "alpha": alpha, "n_comparisons": res["n"],
            "n_raw_discoveries": sum(1 for a in items if a["raw_discovery"]),
            "n_after_correction": res["n_rejected"], "threshold": res["threshold"],
            "by_source": {s: sum(1 for a in items if a["source"] == s)
                          for s in ("claim", "tournament")}, "items": items}


def correct_claims(records: list[dict], *, alpha: float = 0.05, method: str = "bh") -> dict:
    """Apply family-wise correction over runs that carry a p_value.

    Family = all comparisons with a p_value (directional or not). A run is a raw discovery if it
    individually excluded zero (status accepted/rejected). After correction, `corrected[i]` is
    True only if the family-wise test rejects the null for that run.
    Returns the per-run annotations + a summary; runs without a p_value are reported separately.
    """
    fam = [r for r in records if isinstance(r.get("p_value"), (int, float))]
    no_p = [r.get("run_id") for r in records if not isinstance(r.get("p_value"), (int, float))]
    pvals = [float(r["p_value"]) for r in fam]
    res = (bonferroni if method == "bonferroni" else benjamini_hochberg)(pvals, alpha)
    annotated = []
    for r, q, rej in zip(fam, res["qvalues"], res["rejected"]):
        raw = bool(r.get("status") in ("accepted", "rejected"))
        annotated.append({"run_id": r.get("run_id"), "status": r.get("status"),
                          "mean_delta": r.get("mean_delta"), "p_value": r.get("p_value"),
                          "q_value": round(q, 4), "raw_discovery": raw,
                          "survives_correction": bool(rej)})
    return {
        "method": res.get("method"), "alpha": alpha, "n_comparisons": res["n"],
        "n_raw_discoveries": sum(1 for a in annotated if a["raw_discovery"]),
        "n_after_correction": res["n_rejected"],
        "threshold": res["threshold"],
        "runs": annotated, "runs_without_pvalue": no_p,
    }
