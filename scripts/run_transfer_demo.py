"""K1 demo: cross-organism transfer-of-conclusions, end to end (no API).

1. Establish a finding on E. coli (cnn vs rf) via the harness -> a real verdict.json.
2. Run the SAME comparison on yeast as a transfer/replication run that links back to (1).
3. Print the concordance verdict: does the E. coli trend replicate on yeast?

Bounded for speed (few series / repeats); this is a wiring demonstration, not a full study.

Usage: python scripts/run_transfer_demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from orchestration import execution_harness as H  # noqa: E402
from seq2yield.experiments.run_spec import AcceptancePolicy, RunSpec  # noqa: E402


def main() -> int:
    # 1. E. coli source finding (bounded): cnn vs rf at the largest size
    src = RunSpec(
        run_id="transfer-demo-ecoli-cnn-vs-rf", dataset="ecoli",
        intervention_type="model_architecture", model_family="cnn",
        train_sizes=[1000, 2000], iterations=[1, 2], n_series=8, seed=1, maturity_tier="tier_1",
        acceptance_policy=AcceptancePolicy(track="performance", baseline_run_id="2026-06-04-full56",
                                           baseline_model="rf", comparison_train_size=2000))
    print("[1/2] E. coli source: cnn vs rf (bounded) ...")
    sv = H.run(src, changed_files=[], run_tests=False)
    sc = sv["comparison"]
    print(f"   E. coli: status={sv['status']} delta={sc['mean_delta']:.4f} "
          f"CI={[round(x,3) for x in sc['paired_bootstrap_ci']]} "
          f"excl0={sc['ci_excludes_zero']} unit={sc['bootstrap_unit']}")

    # 2. yeast replication linking back to the source
    tgt = RunSpec(
        run_id="transfer-demo-yeast-cnn-vs-rf-xfer", dataset="yeast",
        intervention_type="model_architecture", model_family="cnn",
        transfer_of_run_id=src.run_id, transfer_source_dataset="ecoli",
        train_sizes=[1000, 2000], iterations=[1], seed=1, maturity_tier="tier_1",
        acceptance_policy=AcceptancePolicy(track="performance", baseline_run_id="yeast-baseline",
                                           baseline_model="rf", comparison_train_size=2000))
    print("[2/2] yeast replication (transfer of conclusions) ...")
    tv = H.run(tgt, changed_files=[], run_tests=False)
    tc = tv["comparison"]
    print(f"   yeast: status={tv['status']} delta={tc['mean_delta']:.4f} "
          f"CI={[round(x,3) for x in tc['paired_bootstrap_ci']]} "
          f"excl0={tc['ci_excludes_zero']} unit={tc['bootstrap_unit']}")

    t = tc["transfer"]
    print("\n=== CROSS-ORGANISM TRANSFER VERDICT ===")
    print(f"  verdict: {t['verdict'].upper()}")
    print(f"  reason : {t['reason']}")
    print(f"  source (E. coli, {t['source']['bootstrap_unit']}): delta={t['source']['mean_delta']}")
    print(f"  target (yeast, {t['target']['bootstrap_unit']}): delta={round(t['target']['mean_delta'],4)}")
    if "crossover_agreement" in t:
        print(f"  data-efficiency trend agreement: {t['crossover_agreement']}")
    print("\n(Conclusion-transfer, not weight-transfer: 96nt vs 80nt inputs differ. CIs are never "
          "pooled across organisms — concordance compares sign + significance of two independent "
          "effects.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
