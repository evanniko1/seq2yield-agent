"""Cross-organism transfer-of-conclusions validation (K1).

Direct weight transfer is impossible here (E. coli 96 nt vs yeast 80 nt one-hot dims differ),
so "transfer" means: does a finding ESTABLISHED on one dataset REPLICATE on the other? This is
exactly the generalizability question the paper's cross-dataset claims are about.

`concordance` compares a yeast replication result against the source (E. coli) finding and
returns a structured replication verdict. It NEVER pools CIs across organisms (the bootstrap
units differ — per-series vs per-sequence, C3); it compares the SIGN and SIGNIFICANCE of the
two independently-estimated effects, plus the data-efficiency crossover direction when present.
"""
from __future__ import annotations


def _sign(x: float | None) -> int:
    if x is None:
        return 0
    return 1 if x > 0 else (-1 if x < 0 else 0)


def concordance(source_cmp: dict, target_cmp: dict) -> dict:
    """Replication verdict for a target (e.g. yeast) result vs a source (e.g. E. coli) finding.

    verdict:
      concordant   - both effects significant (CI excludes 0) and same sign -> trend replicates
      discordant   - both significant but opposite sign -> trend reverses across organisms
      inconclusive - either side's CI includes 0 -> cannot confirm the trend transfers
    """
    s_delta, s_sig = source_cmp.get("mean_delta"), bool(source_cmp.get("ci_excludes_zero"))
    t_delta, t_sig = target_cmp.get("mean_delta"), bool(target_cmp.get("ci_excludes_zero"))
    ss, ts = _sign(s_delta), _sign(t_delta)

    if not s_sig:
        verdict = "inconclusive"
        reason = "source finding is not significant (CI includes 0); nothing firm to replicate"
    elif not t_sig:
        verdict = "inconclusive"
        reason = "target effect not significant (CI includes 0); trend neither confirmed nor refuted"
    elif ss == ts and ss != 0:
        verdict = "concordant"
        reason = f"both significant and same direction (sign={ss}); the trend replicates"
    else:
        verdict = "discordant"
        reason = f"both significant but opposite direction (source={ss}, target={ts}); trend reverses"

    out = {
        "verdict": verdict,
        "reason": reason,
        "source": {"mean_delta": s_delta, "ci_excludes_zero": s_sig,
                   "candidate_model": source_cmp.get("candidate_model"),
                   "baseline_model": source_cmp.get("baseline_model"),
                   "bootstrap_unit": source_cmp.get("bootstrap_unit")},
        "target": {"mean_delta": t_delta, "ci_excludes_zero": t_sig,
                   "candidate_model": target_cmp.get("candidate_model"),
                   "baseline_model": target_cmp.get("baseline_model"),
                   "bootstrap_unit": target_cmp.get("bootstrap_unit")},
        "same_sign": ss == ts and ss != 0,
    }

    # data-efficiency: do the crossover directions agree (trend, not just endpoint)?
    s_cross = (source_cmp.get("crossover") or {}).get("trend")
    t_cross = (target_cmp.get("crossover") or {}).get("trend")
    if s_cross and t_cross:
        out["crossover_agreement"] = {"source_trend": s_cross, "target_trend": t_cross,
                                      "agrees": s_cross == t_cross}
    return out
