"""Execution harness — the trusted orchestrator (docs/PROJECT_SPEC.md §10, AGENTS.md).

Runs the acceptance gates in order over a validated RunSpec and emits a verdict
(accepted | rejected | inconclusive) plus a run-card and an audit trail. The harness is more
trusted than any agent: it owns the gates and the final status.

Gate order:
  1. validate RunSpec (schema + tier + files + metric + seed policy)
  2. protected-file guard over the change set (deny-by-default)
  3. tests (pytest) must pass
  4. execute the experiment (single model family)
  5. compare candidate vs baseline (paired bootstrap + acceptance policy)
  6. persist run-card + audit log; (revert handled by patch_manager in the patch loop, M6)
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from seq2yield.experiments import compare as compare_mod  # noqa: E402
from seq2yield.experiments.run_spec import RunSpec, validate_runspec  # noqa: E402
from seq2yield.experiments.runner import per_series_r2, run_runspec  # noqa: E402
from seq2yield.training.reproducibility import environment  # noqa: E402

from . import audit_log, git_guard  # noqa: E402


def _unlocked_tier() -> str:
    cfg = yaml.safe_load((ROOT / "configs/maturity_tiers.yaml").read_text(encoding="utf-8"))
    return cfg["maturity_tiers"].get("unlocked_tier", "tier_0")


def _verdict(run_dir: Path, status: str, detail: dict) -> dict:
    out = {"run_id": detail.get("run_id"), "status": status, **detail}
    (run_dir / "verdict.json").write_text(__import__("json").dumps(out, indent=2), encoding="utf-8")
    audit_log.append(run_dir, "verdict", {"status": status})
    return out


def run(spec: RunSpec, *, changed_files=None, human_review: bool = False,
        run_tests: bool = True) -> dict:
    run_dir = ROOT / "experiments/runs" / spec.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    audit_log.append(run_dir, "received_runspec", {"run_id": spec.run_id,
                                                   "model_family": spec.model_family,
                                                   "tier": spec.maturity_tier})

    # 1. validate RunSpec
    vres = validate_runspec(spec, unlocked_tier=_unlocked_tier())
    audit_log.append(run_dir, "validate_runspec", vres.model_dump())
    if not vres.ok:
        return _verdict(run_dir, "rejected",
                        {"run_id": spec.run_id, "stage": "validation", "reasons": vres.errors})

    # 2. protected-file guard
    paths = git_guard.changed_paths() if changed_files is None else list(changed_files)
    guard = git_guard.check_paths(paths, human_review=human_review,
                                  allowed_files=spec.allowed_files)
    audit_log.append(run_dir, "protected_guard", guard)
    (run_dir / "protected_file_check.json").write_text(
        __import__("json").dumps(guard, indent=2), encoding="utf-8")
    if not guard["passed"]:
        return _verdict(run_dir, "rejected",
                        {"run_id": spec.run_id, "stage": "protected_guard",
                         "reasons": [f"protected-file violations: {guard['violations']}"]})

    # 3. tests
    if run_tests:
        proc = subprocess.run([sys.executable, "-m", "pytest", "-q", str(ROOT / "tests")],
                              capture_output=True, text=True)
        (run_dir / "test_log.txt").write_text(proc.stdout + proc.stderr, encoding="utf-8")
        audit_log.append(run_dir, "tests", {"returncode": proc.returncode})
        if proc.returncode != 0:
            return _verdict(run_dir, "rejected",
                            {"run_id": spec.run_id, "stage": "tests",
                             "reasons": ["pytest failed; see test_log.txt"]})

    # 4. execute experiment
    audit_log.append(run_dir, "execute_start", {})
    result = run_runspec(spec)
    result["metrics"].to_csv(run_dir / "metrics.csv", index=False)
    audit_log.append(run_dir, "execute_done", {"n_rows": int(len(result["metrics"]))})

    # split integrity
    if spec.split_hash and result["split_hash"] != spec.split_hash and \
            spec.acceptance_policy.requires_no_split_change:
        return _verdict(run_dir, "rejected",
                        {"run_id": spec.run_id, "stage": "split_check",
                         "reasons": ["split_hash changed vs RunSpec"]})

    # 5. compare vs baseline (performance track)
    pol = spec.acceptance_policy
    if pol.track != "performance":
        return _verdict(run_dir, "inconclusive",
                        {"run_id": spec.run_id, "stage": "compare",
                         "reasons": [f"track '{pol.track}' comparison not implemented in M3"]})

    base_csv = ROOT / "experiments/runs" / pol.baseline_run_id / "metrics.csv"
    if not base_csv.exists():
        return _verdict(run_dir, "rejected",
                        {"run_id": spec.run_id, "stage": "compare",
                         "reasons": [f"baseline metrics not found: {base_csv}"]})
    base_df = pd.read_csv(base_csv)

    size = pol.comparison_train_size or max(
        set(spec.train_sizes) & set(base_df["train_size"].unique()))
    cand_ps = per_series_r2(result["metrics"], size, spec.model_family)
    base_ps = per_series_r2(base_df, size, pol.baseline_model)

    cmp = compare_mod.compare(base_ps, cand_ps, pol, seed=spec.seed)
    cmp["comparison_train_size"] = size
    cmp["baseline_model"] = pol.baseline_model
    cmp["candidate_model"] = spec.model_family

    # per-size statistical verdicts + crossover for multi-size sweeps (rigorous answer to
    # "at what N does the candidate catch up?")
    sizes = sorted(set(spec.train_sizes) & set(base_df["train_size"].unique()))
    if len(sizes) > 1:
        per_size = compare_mod.compare_per_size(base_df, result["metrics"], sizes,
                                                pol.baseline_model, spec.model_family, pol,
                                                seed=spec.seed)
        cmp["per_size"] = per_size
        cmp["crossover"] = compare_mod.crossover_analysis(per_size)
    audit_log.append(run_dir, "compare", cmp)

    return _verdict(run_dir, cmp["status"],
                    {"run_id": spec.run_id, "stage": "complete", "comparison": cmp,
                     "environment": environment(), "warnings": vres.warnings})
