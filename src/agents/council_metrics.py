"""Council economics + reviewer-calibration metrics (RESEARCH_AGENDA II-6, II-7).

II-6 cost-per-accepted-claim: what does one durable scientific claim cost in model spend?
II-7 reviewer discrimination: do a reviewer's scores separate accepted from rejected runs (i.e. are
its scores CALIBRATED against the realized harness outcome, not just internally consistent)?
Both are read-model joins — no new instrumentation.
"""
from __future__ import annotations

from statistics import mean


def cost_per_claim(con) -> dict:
    """Total model spend / number of accepted claims (+ per-status counts). `con` = store.connect()."""
    from orchestration import store
    cost = store.cost_summary(con)
    n_claims = len(store.accepted_claims(con))
    rows = con.execute("SELECT status, COUNT(*) n FROM claim GROUP BY status").fetchall()
    by_status = {r["status"]: r["n"] for r in rows}
    return {"total_cost_usd": cost["total_cost_usd"], "n_calls": cost["n_calls"],
            "n_accepted_claims": n_claims,
            "cost_per_accepted_claim": (round(cost["total_cost_usd"] / n_claims, 4)
                                        if n_claims else None),
            "by_status": by_status}


def reviewer_discrimination(cycles: list[dict]) -> dict:
    """Per role, the mean 'overall' score it gave to ACCEPTED vs REJECTED runs and the gap. A
    positive gap = the reviewer's scores discriminate good from bad (calibrated); ≈0 = its scores
    carry no outcome signal. cycles: [{"outcome": "accepted"|"rejected",
    "reviews": [{role, score_feasibility, score_scientific_value, score_confoundedness,
    score_reproducibility}]}]."""
    acc: dict[str, list[float]] = {}
    rej: dict[str, list[float]] = {}
    for c in cycles:
        bucket = acc if c.get("outcome") == "accepted" else (rej if c.get("outcome") == "rejected" else None)
        if bucket is None:
            continue
        for r in c.get("reviews", []):
            overall = sum(r.get(k, 3) for k in ("score_feasibility", "score_scientific_value",
                                                "score_confoundedness", "score_reproducibility"))
            bucket.setdefault(r.get("role"), []).append(overall)
    out = {}
    for role in sorted(set(acc) | set(rej)):
        ma = round(mean(acc[role]), 3) if acc.get(role) else None
        mr = round(mean(rej[role]), 3) if rej.get(role) else None
        gap = round(ma - mr, 3) if (ma is not None and mr is not None) else None
        out[role] = {"mean_on_accepted": ma, "mean_on_rejected": mr, "discrimination_gap": gap,
                     "calibrated": (gap is not None and gap > 0)}
    return out
