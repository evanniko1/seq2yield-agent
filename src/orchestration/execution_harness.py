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


def _attach_diagnostics(spec, cmp: dict, run_dir: Path) -> None:
    """K4: attach methodology diagnostics + flags to the comparison (ADVISORY — never changes the
    verdict). Wrapped so a diagnostic failure can never sink a real run."""
    try:
        from seq2yield.diagnostics import collect
        size = cmp.get("comparison_train_size") or max(spec.train_sizes)
        diag = collect.diagnose(spec, size, per_size=cmp.get("per_size"))
        cmp["diagnostics"] = diag["diagnostics"]
        cmp["methodology_flags"] = diag["methodology_flags"]
        cmp["flag_summary"] = diag["flag_summary"]
        audit_log.append(run_dir, "diagnostics", diag["flag_summary"])
    except Exception as e:  # advisory only
        cmp["diagnostics_error"] = str(e)[:200]
        audit_log.append(run_dir, "diagnostics_error", {"error": str(e)[:200]})


def _verdict(run_dir: Path, status: str, detail: dict) -> dict:
    out = {"run_id": detail.get("run_id"), "status": status, **detail}
    (run_dir / "verdict.json").write_text(__import__("json").dumps(out, indent=2), encoding="utf-8")
    audit_log.append(run_dir, "verdict", {"status": status})
    return out


def _baseline_spec(spec: RunSpec, baseline_model: str) -> RunSpec:
    """Build the in-run baseline: identical to the candidate EXCEPT the one varied knob is reset
    to its default, so the comparison isolates exactly that knob (controlled comparison)."""
    b = spec.model_copy(deep=True)
    it = spec.intervention_type
    if it in ("model_architecture", "data_efficiency"):
        b.model_family = baseline_model            # different model; keep scope/features/etc.
    elif it == "feature_representation":
        b.feature_set = "one_hot"                  # same model, scaling kept -> isolate feature
    elif it == "sampling_design":
        b.sampling_policy = "random"
    elif it == "feature_scaling":
        b.feature_scaling = "none"
    elif it == "training_procedure":
        b.hyperparameters = {}
    else:
        b.model_family = baseline_model
    return b


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
    if _is_pooled(spec.dataset):                        # K6: any pooled dataset -> sequence-level path
        return _run_pooled(spec, run_dir, vres)
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

    # Choose the baseline reference. The registry (per-series, one_hot, random, defaults,
    # unscaled) is a valid control ONLY for a plain different-model comparison; any other axis
    # must be compared to an in-run baseline that is identical to the candidate EXCEPT for the
    # one varied knob — otherwise the comparison is confounded (CRITIQUE / audit).
    plain = (spec.scope == "global" and spec.feature_set == "one_hot"
             and spec.sampling_policy == "random" and spec.feature_scaling == "none"
             and not spec.hyperparameters)
    use_registry = plain and spec.intervention_type in ("model_architecture", "data_efficiency")
    if use_registry:
        base_csv = ROOT / "experiments/runs" / pol.baseline_run_id / "metrics.csv"
        if not base_csv.exists():
            return _verdict(run_dir, "rejected",
                            {"run_id": spec.run_id, "stage": "compare",
                             "reasons": [f"baseline metrics not found: {base_csv}"]})
        base_df = pd.read_csv(base_csv)
        baseline_source = "registry:" + pol.baseline_run_id
    else:
        base_df = run_runspec(_baseline_spec(spec, pol.baseline_model))["metrics"]
        baseline_source = "in_run"

    size = pol.comparison_train_size or max(
        set(spec.train_sizes) & set(base_df["train_size"].unique()))
    cand_ps = per_series_r2(result["metrics"], size, spec.model_family)
    base_ps = per_series_r2(base_df, size, pol.baseline_model)

    cmp = compare_mod.compare(base_ps, cand_ps, pol, seed=spec.seed)
    cmp["comparison_train_size"] = size
    cmp["baseline_model"] = pol.baseline_model
    cmp["candidate_model"] = spec.model_family
    cmp["baseline_source"] = baseline_source
    # C3: label the bootstrap UNIT. E. coli comparisons resample per-series R² (the unit is a
    # mutational series); the yeast benchmark resamples test sequences. CIs across different
    # units are NOT directly comparable — recording the unit prevents silent cross-claims.
    cmp["bootstrap_unit"] = "series"
    # capacity transparency: parameter counts for torch models (None for sklearn). Flags
    # un-controlled-capacity architecture comparisons (CRITIQUE C5).
    from seq2yield.models import registry as _reg
    cp, bp = _reg.param_count(spec.model_family), _reg.param_count(pol.baseline_model)
    cmp["candidate_params"], cmp["baseline_params"] = cp, bp
    if cp and bp:
        cmp["param_ratio"] = round(cp / bp, 2)
        cmp["param_fairness"] = "ok" if 0.5 <= cp / bp <= 2.0 else "capacity-imbalanced"
    # per-series heterogeneity: where the winner differs across series (Q6)
    cmp["heterogeneity"] = compare_mod.heterogeneity_analysis(base_ps, cand_ps)

    # per-size statistical verdicts + crossover for multi-size sweeps (rigorous answer to
    # "at what N does the candidate catch up?")
    sizes = sorted(set(spec.train_sizes) & set(base_df["train_size"].unique()))
    if len(sizes) > 1:
        per_size = compare_mod.compare_per_size(base_df, result["metrics"], sizes,
                                                pol.baseline_model, spec.model_family, pol,
                                                seed=spec.seed)
        cmp["per_size"] = per_size
        cmp["crossover"] = compare_mod.crossover_analysis(per_size)
    _attach_diagnostics(spec, cmp, run_dir)            # K4: advisory methodology flags
    audit_log.append(run_dir, "compare", cmp)

    return _verdict(run_dir, cmp["status"],
                    {"run_id": spec.run_id, "stage": "complete", "comparison": cmp,
                     "environment": environment(), "warnings": vres.warnings})


def _is_pooled(dataset_id: str) -> bool:
    from seq2yield.data import datasets
    return datasets.exists(dataset_id) and datasets.spec(dataset_id).structure == "pooled"


def _run_pooled(spec: RunSpec, run_dir: Path, vres) -> dict:
    """K6: generic pooled path (any `structure: pooled` dataset). Trains candidate + in-run baseline
    pooled, evaluates on a fixed held-out set with a SEQUENCE-LEVEL paired bootstrap
    (bootstrap_unit=sequence, C3 — never pooled with per-series CIs). If the run links a source
    finding (transfer_of_run_id), attaches a cross-dataset concordance verdict (replication)."""
    import json

    from seq2yield.experiments import pooled_runner, transfer
    from seq2yield.statistics.bootstrap import paired_bootstrap_r2
    from seq2yield.training import metrics as M

    pol = spec.acceptance_policy
    if pol.track != "performance":
        return _verdict(run_dir, "inconclusive",
                        {"run_id": spec.run_id, "stage": "compare",
                         "reasons": [f"track '{pol.track}' not implemented for pooled datasets"]})

    cand = pooled_runner.run_pooled(spec)
    base_spec = _baseline_spec(spec, pol.baseline_model)
    base = pooled_runner.run_pooled(base_spec, model_family=base_spec.model_family)
    y = cand["y_test"]
    sizes = sorted(set(cand["preds"]) & set(base["preds"]))
    audit_log.append(run_dir, "execute_done", {"n_test": int(len(y)), "sizes": sizes})

    rows = []
    for size in sizes:
        rows.append({"dataset": spec.dataset, "train_size": size, "model": spec.model_family,
                     "r2": M.r2(y, cand["preds"][size])})
        rows.append({"dataset": spec.dataset, "train_size": size, "model": pol.baseline_model,
                     "r2": M.r2(y, base["preds"][size])})
    pd.DataFrame(rows).to_csv(run_dir / "metrics.csv", index=False)

    per_size = []
    for size in sizes:
        pb = paired_bootstrap_r2(y, cand["preds"][size], base["preds"][size], seed=spec.seed)
        status, reasons = compare_mod._decide(pb["mean_delta"], pb["excludes_zero"], pb["ci"], pol)
        per_size.append({"train_size": int(size), "status": status,
                         "candidate_mean": float(M.r2(y, cand["preds"][size])),
                         "baseline_mean": float(M.r2(y, base["preds"][size])),
                         "mean_delta": float(pb["mean_delta"]), "paired_bootstrap_ci": pb["ci"],
                         "ci_excludes_zero": pb["excludes_zero"], "p_value": pb["p_value"],
                         "reasons": reasons})

    size = pol.comparison_train_size if pol.comparison_train_size in sizes else max(sizes)
    cmp = dict(next(p for p in per_size if p["train_size"] == size))
    cmp.update({"comparison_train_size": size, "candidate_model": spec.model_family,
                "baseline_model": pol.baseline_model, "baseline_source": "in_run_pooled",
                "bootstrap_unit": "sequence", "dataset": spec.dataset, "n_test": int(len(y))})
    if len(per_size) > 1:
        cmp["per_size"] = per_size
        cmp["crossover"] = compare_mod.crossover_analysis(per_size)

    # cross-organism transfer: replicate a prior (E. coli) finding and judge concordance
    if spec.transfer_of_run_id:
        src = ROOT / "experiments/runs" / spec.transfer_of_run_id / "verdict.json"
        if src.exists():
            source_cmp = json.loads(src.read_text(encoding="utf-8")).get("comparison", {})
            cmp["transfer"] = transfer.concordance(source_cmp, cmp)
            cmp["transfer"]["source_run_id"] = spec.transfer_of_run_id
            cmp["transfer"]["source_dataset"] = spec.transfer_source_dataset or "ecoli"
        else:
            cmp["transfer"] = {"verdict": "inconclusive",
                               "reason": f"source run {spec.transfer_of_run_id} not found"}

    _attach_diagnostics(spec, cmp, run_dir)            # K4: advisory methodology flags
    audit_log.append(run_dir, "compare", cmp)
    return _verdict(run_dir, cmp["status"],
                    {"run_id": spec.run_id, "stage": "complete", "comparison": cmp,
                     "environment": environment(), "warnings": vres.warnings})
