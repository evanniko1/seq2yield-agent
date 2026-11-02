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
from seq2yield.experiments import runner as runner_mod  # noqa: E402
from seq2yield.experiments.runner import per_series_r2, run_runspec  # noqa: E402
from seq2yield.training.reproducibility import environment  # noqa: E402

from . import audit_log, baseline_guard, git_guard  # noqa: E402


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


def _flag_gate(cmp: dict, run_dir: Path) -> None:
    """(b) HARD methodology gate. Most diagnostic flags stay advisory, but flags marked
    `blocking: true` in configs/methodology_pitfalls.yaml (e.g. train/test leakage) mean the
    comparison itself can't be trusted — so an ACCEPTED result is withheld: downgraded to
    inconclusive with the blocking flag ids recorded. This only ever WITHHOLDS a claim; it never
    upgrades a verdict, and it leaves rejected/inconclusive runs unchanged. If the diagnostic probe
    itself failed (no flags computed) the gate is a no-op — the diagnostics_error is already logged."""
    if cmp.get("status") != "accepted":
        return
    blocking = [f for f in cmp.get("methodology_flags", []) if f.get("blocking")]
    if not blocking:
        return
    cmp["status"] = "inconclusive"
    cmp["gated_by_flags"] = [f["id"] for f in blocking]
    cmp.setdefault("reasons", []).append(
        "accepted result withheld — blocking methodology flag(s): "
        + ", ".join(f"{f['id']}({f['severity']})" for f in blocking))
    audit_log.append(run_dir, "flag_gate",
                     {"downgraded": "accepted->inconclusive", "flags": cmp["gated_by_flags"]})


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


def _baseline_choice(spec: RunSpec, pol) -> tuple[bool, dict | None]:
    """May this comparison use the STORED baseline registry as its control?

    Two independent conditions, both of which have bitten this project:
    1. the comparison must be *plain* (one_hot / random / unscaled / no hyperparameters / global
       scope) — the registry was built that way, so any other axis is confounded against it;
    2. the registry's provenance must still match this checkout (ML-02) — a stored result measured
       with different model code is not a control, it is a different experiment.
    Returns (use_registry, provenance_decision | None); None means the registry was never eligible,
    so there is no provenance decision to record."""
    plain = (spec.scope == "global" and spec.feature_set == "one_hot"
             and spec.sampling_policy == "random" and spec.feature_scaling == "none"
             and not spec.hyperparameters)
    if not (plain and spec.intervention_type in ("model_architecture", "data_efficiency")):
        return False, None
    prov = baseline_guard.check(pol.baseline_run_id, [spec.model_family, pol.baseline_model])
    # 3. (audit ML-01b) a multi-seed candidate needs a multi-seed control. The stored registry held
    # ONE R² per (series, size, model) from a single baseline seed, so pairing a 3-seed candidate
    # against it would put the candidate's seed variance in the interval and the baseline's
    # nowhere — and the baseline's is not zero, it is unmeasured. Same remedy as ML-02's: refuse
    # the registry and train an in-run control at the SAME seeds, which is a correct comparison
    # regardless of what the registry happens to contain.
    seeds = runner_mod.model_seeds(spec)
    if len(seeds) > 1:
        ok_seeds, why = _registry_seed_coverage(pol.baseline_run_id, seeds)
        if not ok_seeds:
            prov = dict(prov)
            prov["ok"] = False
            prov["violations"] = list(prov.get("violations") or []) + ["model_seed_coverage"]
            prov["reason"] = " | ".join(x for x in (prov.get("reason"), why) if x)
    return bool(prov["ok"]), prov


def _registry_seed_coverage(baseline_run_id: str, seeds) -> tuple[bool, str]:
    """Does the stored registry hold rows for every model-init seed this run uses?"""
    csv = ROOT / "experiments/runs" / baseline_run_id / "metrics.csv"
    if not csv.exists():
        return False, f"registry metrics.csv not found: {csv}"
    head = pd.read_csv(csv, nrows=1)
    if "model_seed" not in head.columns:
        return False, (f"registry '{baseline_run_id}' has no model_seed column: it was built from a "
                       f"single baseline seed and cannot be paired seed-matched against "
                       f"{len(set(seeds))} candidate seeds (audit ML-01b)")
    have = set(pd.read_csv(csv, usecols=["model_seed"])["model_seed"].astype(int).unique())
    missing = sorted(set(int(s) for s in seeds) - have)
    if missing:
        return False, (f"registry '{baseline_run_id}' is missing model_seed {missing} "
                       f"(has {sorted(have)})")
    return True, ""


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
    else:                                               # caller skipped the gate (e.g. no-patch fast)
        audit_log.append(run_dir, "tests", {"skipped": True,
                                            "reason": "run_tests=False (no changeset to validate)"})

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
    use_registry, prov = _baseline_choice(spec, pol)
    if prov is not None:
        audit_log.append(run_dir, "baseline_guard", prov)
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
        if prov is not None:
            baseline_source = f"in_run (registry refused: {','.join(prov['violations'])})"

    # int() is load-bearing: `base_df["train_size"].unique()` yields numpy int64, and a set
    # intersection of Python ints with numpy ints can return either type, so this came out as an
    # int64 that `audit_log.append` could not serialize — killing the run at the point where it
    # writes its own trail.
    size = int(pol.comparison_train_size or max(
        set(spec.train_sizes) & set(base_df["train_size"].unique())))
    cand_ps = per_series_r2(result["metrics"], size, spec.model_family)
    base_ps = per_series_r2(base_df, size, pol.baseline_model)
    # ML-01b: the interval comes from the (series × seed) grid whenever both arms have one, so the
    # model-init component is inside it. `*_ps` stays the seed-averaged point estimate, which is
    # what the heterogeneity report wants.
    cand_by = runner_mod.per_series_r2_by_seed(result["metrics"], size, spec.model_family)
    base_by = runner_mod.per_series_r2_by_seed(base_df, size, pol.baseline_model)
    if len(cand_by) > 1 and len(base_by) > 1:
        cmp = compare_mod.compare_multiseed(base_by, cand_by, pol, seed=spec.seed)
    else:
        cmp = compare_mod.compare(base_ps, cand_ps, pol, seed=spec.seed)
        cmp["n_seeds"] = {"candidate": len(cand_by), "baseline": len(base_by)}
        cmp["reasons"] = cmp["reasons"] + [
            f"single-seed arm (candidate {len(cand_by)} seed(s), baseline {len(base_by)}): the "
            "model-init variance component is NOT in this interval (audit ML-01b)"]
    cmp["comparison_train_size"] = size
    # `per_series_r2_by_seed` keys a frame with no `model_seed` column as -1 ("one seed, identity
    # unknown"), which is right for the estimator and cryptic in a record. The candidate's seeds
    # are known from the run; a pre-2026-11-02 registry's are not, and null says so.
    cmp["model_seeds"] = {
        "candidate": (result["model_seeds"] if sorted(cand_by) == [-1] else sorted(cand_by)),
        "baseline": (None if sorted(base_by) == [-1] else sorted(base_by)),
    }
    # SA-2: the selection RULE travels with the claim, and so does how many series existed to
    # choose from — "10 series" means something different when it is a sample than when it is the
    # sorted first ten.
    cmp["series_selection"] = result["series_selection"]
    cmp["series_sample_seed"] = result["series_sample_seed"]
    cmp["n_series_available"] = result["n_series_available"]
    cmp["series_used"] = list(result["series"])
    cmp["baseline_model"] = pol.baseline_model
    cmp["candidate_model"] = spec.model_family
    cmp["baseline_source"] = baseline_source
    if prov is not None:                      # the provenance decision travels on the run-card
        cmp["baseline_provenance"] = {k: prov[k] for k in
                                      ("ok", "reason", "violations", "drift", "registry_run_id")}
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
    _flag_gate(cmp, run_dir)                            # (b): blocking flags withhold an accept
    audit_log.append(run_dir, "compare", cmp)

    return _verdict(run_dir, cmp["status"],
                    {"run_id": spec.run_id, "stage": "complete", "comparison": cmp,
                     "environment": environment(), "warnings": vres.warnings})


def _is_pooled(dataset_id: str) -> bool:
    from seq2yield.data import datasets
    return datasets.exists(dataset_id) and datasets.spec(dataset_id).structure == "pooled"


def _run_pooled(spec: RunSpec, run_dir: Path, vres) -> dict:
    """K6: generic pooled path (any `structure: pooled` dataset). Trains candidate + in-run baseline
    pooled, evaluates on a fixed held-out set with a paired bootstrap whose unit is the design
    FAMILY when the dataset declares one and the test sequence otherwise (recorded as the C3
    `bootstrap_unit` — never pooled with per-series CIs). If the run links a source finding
    (transfer_of_run_id), attaches a cross-dataset concordance verdict (replication)."""
    import json

    from seq2yield.experiments import pooled_runner, transfer
    from seq2yield.statistics.bootstrap import paired_bootstrap_r2, paired_r2_multiseed
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

    # The resampling unit: whole design families when the dataset declares them, else one test
    # sequence. A family is ONE draw repeated, so resampling its members as if independent gives
    # a CI that is too narrow (audit ST-7). `unit` travels onto the run-card as bootstrap_unit.
    groups = cand.get("test_families")
    per_size, unit, n_units = [], "sequence", int(len(y))
    for size in sizes:
        # Seed AND unit variance (audit ML-01). Each arm was fitted once per model-init seed with
        # the split and subsample held fixed, so the seed component is estimable; with one seed
        # the estimator says so in a note rather than silently reporting the narrow interval.
        cbs, bbs = cand.get("preds_by_seed") or {}, base.get("preds_by_seed") or {}
        if cbs.get(size) and bbs.get(size):
            pb = paired_r2_multiseed(y, cbs[size], bbs[size], groups=groups, seed=spec.seed)
        else:
            pb = paired_bootstrap_r2(y, cand["preds"][size], base["preds"][size], groups=groups,
                                     seed=spec.seed)
        unit, n_units = pb["unit"], pb["n_units"]
        status, reasons, margin_verdict = compare_mod._decide(
            pb["mean_delta"], pb["excludes_zero"], pb["ci"], pol,
            ci_one_sided=pb.get("ci_one_sided"))
        if pb.get("note"):                      # too few units to bootstrap: say why, don't hide it
            reasons = reasons + [pb["note"]]
        per_size.append({"train_size": int(size), "status": status,
                         "candidate_mean": float(M.r2(y, cand["preds"][size])),
                         "baseline_mean": float(M.r2(y, base["preds"][size])),
                         "mean_delta": float(pb["mean_delta"]), "paired_bootstrap_ci": pb["ci"],
                         "ci_excludes_zero": pb["excludes_zero"], "p_value": pb["p_value"],
                         "ci_method": pb.get("ci_method"), "p_method": pb.get("p_method"),
                         "p_floor": pb.get("p_floor"), "ci_one_sided": pb.get("ci_one_sided"),
                         "margin": pol.min_delta_r2, "margin_verdict": margin_verdict,
                         # ML-01: how much of the uncertainty is the model-init seed
                         "n_seeds": pb.get("n_seeds"), "seed_deltas": pb.get("seed_deltas"),
                         "seed_spread": pb.get("seed_spread"), "se_seed": pb.get("se_seed"),
                         "se_unit": pb.get("se_unit"),
                         "reasons": reasons})

    size = pol.comparison_train_size if pol.comparison_train_size in sizes else max(sizes)
    cmp = dict(next(p for p in per_size if p["train_size"] == size))
    cmp.update({"comparison_train_size": size, "candidate_model": spec.model_family,
                "baseline_model": pol.baseline_model, "baseline_source": "in_run_pooled",
                "bootstrap_unit": unit, "n_bootstrap_units": int(n_units),
                "model_seeds": cand.get("model_seeds"),
                "dataset": spec.dataset, "n_test": int(len(y))})
    # Record HOW the data was split (strategy, family source, family counts). A grouped-holdout
    # result is not comparable with an older stratified-holdout one, so the split must never be
    # implicit on a run-card (audit SA-1/ML-03).
    cmp["split"] = pooled_runner.split_report(spec)
    if cand.get("test_families") is not None:
        cmp["n_test_families"] = int(len(set(cand["test_families"])))
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
    _flag_gate(cmp, run_dir)                            # (b): blocking flags withhold an accept
    audit_log.append(run_dir, "compare", cmp)
    return _verdict(run_dir, cmp["status"],
                    {"run_id": spec.run_id, "stage": "complete", "comparison": cmp,
                     "environment": environment(), "warnings": vres.warnings})
