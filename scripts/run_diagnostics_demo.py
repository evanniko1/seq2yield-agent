"""K4 demo: methodology diagnostics -> named flags -> critic -> council feedback.

Runs a bounded yeast experiment through the harness (which now attaches deterministic diagnostics
+ advisory flags), narrates the flags with the methodology critic (authority provider if a key is set,
else local Ollama),
and shows how the flags feed back as 'open methodology flags' the council would see next cycle.

Usage: python scripts/run_diagnostics_demo.py
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

from agents import methodology_critic  # noqa: E402
from orchestration import execution_harness as H  # noqa: E402
from seq2yield.experiments.run_spec import AcceptancePolicy, RunSpec  # noqa: E402


def main() -> int:
    spec = RunSpec(
        run_id="k4-demo-yeast-rf-vs-mlp", dataset="yeast", intervention_type="model_architecture",
        model_family="rf", train_sizes=[500, 1000], iterations=[1], seed=1, maturity_tier="tier_1",
        acceptance_policy=AcceptancePolicy(track="performance", baseline_run_id="yeast-baseline",
                                           baseline_model="mlp", comparison_train_size=1000))
    print("[harness] running bounded yeast rf-vs-mlp (diagnostics attach automatically) ...")
    v = H.run(spec, changed_files=[], run_tests=False)
    cmp = v["comparison"]
    d = cmp["diagnostics"]
    print(f"\nstatus={v['status']}  (diagnostics are ADVISORY — they do not change this verdict)")

    print("\n=== OBSERVABLE DIAGNOSTIC SIGNALS (harness-computed, deterministic) ===")
    print(f"  generalization gap : train R²={d['generalization_gap']['train_r2']} "
          f"test R²={d['generalization_gap']['test_r2']} gap={d['generalization_gap']['gap']}")
    print(f"  calibration slope  : {d['calibration']['slope']}  (1.0 = well-calibrated)")
    print(f"  split representativ.: KS={d['representativeness']['ks']} "
          f"mean_shift={d['representativeness']['mean_shift_std']}σ")
    print(f"  sequence leakage   : {d['leakage']['leak_frac']} ({d['leakage']['n_leaked']} dup)")
    print(f"  target extrapolation: {d['coverage']['extrapolated_frac']}")
    print(f"  learning curve      : still_improving={d['learning_curve']['still_improving']}")

    flags = cmp["methodology_flags"]
    print(f"\n=== METHODOLOGY FLAGS ({cmp['flag_summary']['by_severity']}) ===")
    for f in flags:
        print(f"  [{f['severity']}] {f['id']}: {f['description']}")
        print(f"        -> suggested follow-up: {f['intervention_hint']} ({f['suggested']})")

    print("\n=== METHODOLOGY CRITIC (local Ollama, free) ===")
    crit, who = methodology_critic.review(
        d, flags, {"candidate_model": "rf", "baseline_model": "mlp", "dataset": "yeast",
                   "status": v["status"], "mean_delta": cmp["mean_delta"]},
        allow_local_fallback=True)
    print(f"  critic={who} severity={crit.severity}")
    print(f"  summary: {crit.summary}")
    for c in crit.concerns:
        print(f"   - concern: {c}")
    for s in crit.suggested_followups:
        print(f"   - follow-up: {s}")

    # feedback loop: what the council sees next cycle
    rec = [{"methodology_flags": flags}]
    print("\n=== COUNCIL FEEDBACK (open flags surfaced to the next generation cycle) ===")
    for f in methodology_critic.open_flags(rec):
        print(f"  open: {f['id']} -> propose a {f['intervention_hint']} experiment to investigate")

    import shutil
    shutil.rmtree(ROOT / "experiments/runs/k4-demo-yeast-rf-vs-mlp", ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
