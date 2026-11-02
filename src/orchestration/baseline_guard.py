"""Baseline-registry provenance guard (audit ML-02; configs/baseline_registry.yaml).

`git_guard` decides whether a change may be made; this decides whether a STORED RESULT may still
be used as a control. The registry is a CSV of R² values measured months ago: comparing a fresh
candidate against it is only valid while the code that produced those values still produces them.

It stopped being valid silently. RF's `max_features` default changed after the registry was locked,
so an unchanged RF measured −0.0158 R² against its own stored baseline at N=2000 and the harness
reported a rejection; the same drift flipped a published model ranking (docs/RETRACTIONS.md R-2).
Nothing in the pipeline could notice, because nothing recorded what the registry was built with.

The rule here is deliberately narrow, so it is believed rather than disabled: the registry is
refused when a file that can change a predicted number changed, or when the dataset/split hashes
moved — not when any commit happens. A refusal is not an error: the harness trains an in-run
baseline instead, which is always a valid control, and records why it had to.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "baseline_registry.yaml"
# The script that produces the registry: the entry point its provenance scope is derived from.
BUILDER = "scripts/reproduce_baselines.py"
# Below this many files the import walk has clearly failed (the real scope is ~22), and a guard
# that silently narrowed itself to nothing would be worse than a noisy one.
MIN_SCOPE = 8


@lru_cache(maxsize=1)
def config() -> dict:
    if not CONFIG.exists():
        return {"run_id": None, "require": {}}
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8")) or {}


def registry_run_id() -> str | None:
    """The canonical registry id — the single pointer every caller should use."""
    return config().get("run_id")


def registry_path(name: str = "metrics.csv", *, root: Path | None = None) -> Path:
    """Path to a file in the canonical registry (`metrics.csv`, `data_size_curve.csv`, ...).

    Analysis code that reads the registry directly (insight, mixed-effects, ranking notes) should
    resolve it here rather than hardcoding a dated folder: six hardcoded copies are how a stale
    registry survives a code change (ML-02)."""
    return Path(root or ROOT) / "experiments/runs" / (registry_run_id() or "") / name


def _run_card(run_id: str, model: str, root: Path | None = None) -> dict | None:
    card = Path(root or ROOT) / "experiments/runs" / run_id / model / "run_card.json"
    if not card.exists():
        return None
    try:
        return json.loads(card.read_text(encoding="utf-8"))
    except Exception:
        return None


def _current_hashes(root: Path | None = None) -> dict:
    """dataset_hash / split_hash of the data in this checkout (None when absent, e.g. on CI)."""
    base = Path(root or ROOT)
    out: dict = {"dataset_hash": None, "split_hash": None}
    try:
        out["dataset_hash"] = json.loads(
            (base / "data/processed/dataset_version.json").read_text(encoding="utf-8"))["dataset_hash"]
    except Exception:
        pass
    try:
        from seq2yield.data.splits import load_manifest
        out["split_hash"] = load_manifest(base / "data/splits")["split_hash"]
    except Exception:
        pass
    return out


def registry_status(run_id: str | None = None, *, root: Path | None = None) -> dict:
    """Is the registry COMPLETE, not merely present?

    The builder checkpoints `metrics.csv` after every (iteration, series), so a registry under
    construction is a readable file covering an arbitrary subset of series — and a subset is not a
    neutral baseline (the audit's SA-2: the first ten series are easier than average). Run-cards
    are written only at the end and declare the grid that was intended, so they are the manifest
    this checks the rows against."""
    base = Path(root or ROOT)
    run_id = run_id or registry_run_id()
    out = {"ready": False, "reason": "", "run_id": run_id, "n_rows": 0}
    if not run_id:
        out["reason"] = "no registry configured"
        return out
    rdir = base / "experiments/runs" / run_id
    metrics = rdir / "metrics.csv"
    if not metrics.exists():
        out["reason"] = f"{metrics} missing"
        return out
    cards = {p.parent.name: p for p in rdir.glob("*/run_card.json")}
    if not cards:
        out["reason"] = (f"registry '{run_id}' has no run-cards: it is unfinished (the builder "
                         "writes them last) or was never completed")
        return out
    import pandas as pd
    try:
        df = pd.read_csv(metrics)
        assert {"model", "series", "iteration", "train_size"} <= set(df.columns)
    except Exception as e:
        out["reason"] = f"{metrics} is not a readable per-series metrics table ({e})"[:200]
        return out
    out["n_rows"] = int(len(df))
    for model, path in sorted(cards.items()):
        card = json.loads(path.read_text(encoding="utf-8"))
        want_iters = {f"iteration_{i}" for i in (card.get("iterations") or [])}
        want_sizes = {int(s) for s in (card.get("results") or {})}
        sub = df[df["model"] == model]
        have_iters = set(sub["iteration"].unique())
        have_sizes = {int(s) for s in sub["train_size"].unique()}
        n_series = int(sub["series"].nunique())
        expect = card.get("n_series")
        if want_iters - have_iters or want_sizes - have_sizes or (expect and n_series != expect):
            out["reason"] = (f"registry '{run_id}' is incomplete for {model}: "
                             f"{n_series}/{expect} series, iterations {sorted(have_iters)} vs "
                             f"{sorted(want_iters)}, sizes {sorted(have_sizes)} vs "
                             f"{sorted(want_sizes)}")
            return out
        out[model] = {"n_series": n_series, "n_iterations": len(have_iters),
                      "train_sizes": sorted(have_sizes)}
    out["ready"] = True
    out["reason"] = f"registry '{run_id}' is complete ({out['n_rows']} rows)"
    return out


def registry_ready(run_id: str | None = None, *, root: Path | None = None) -> bool:
    """True when the canonical registry is complete enough to read. Analysis code that loads the
    registry directly should check this instead of `Path.exists()`."""
    return registry_status(run_id, root=root)["ready"]


def check(run_id: str | None, models, *, root: Path | None = None) -> dict:
    """May the stored registry `run_id` be used as the control for `models`?

    Returns {ok, reason, violations, registry, current, drift} — `drift` naming the changed model
    defaults when they are recorded on both sides, which is the whole point: "the fingerprint
    changed" is a shrug, "rf.max_features: 1.0 -> 'sqrt'" is an answer.
    """
    from seq2yield.training import provenance
    req = config().get("require") or {}
    models = [m for m in (models if isinstance(models, (list, tuple, set)) else [models]) if m]
    out: dict = {"ok": False, "reason": "", "violations": [], "registry_run_id": run_id,
                 "models": list(models), "drift": []}
    if not run_id:
        out["reason"] = "no baseline registry configured"
        out["violations"] = ["missing_run_id"]
        return out

    cards = {m: _run_card(run_id, m, root) for m in models}
    missing = [m for m, c in cards.items() if c is None]
    if missing:
        out["reason"] = (f"registry '{run_id}' has no run-card for {missing}: nothing records what "
                         "those numbers were produced with")
        out["violations"] = ["missing_run_card"]
        return out

    status = registry_status(run_id, root=root)
    if not status["ready"]:                  # present but truncated: a subset is not a baseline
        out["reason"] = status["reason"]
        out["violations"] = ["incomplete_registry"]
        return out

    cur_num = provenance.numerics_fingerprint()["fingerprint"]
    cur_defaults = provenance.model_defaults()
    cur = _current_hashes(root)
    envs = {m: (c.get("environment") or {}) for m, c in cards.items()}
    reg_num = {m: ((e.get("numerics") or {}).get("fingerprint")) for m, e in envs.items()}

    unstamped = [m for m, f in reg_num.items() if not f]
    if unstamped:
        out["reason"] = (f"registry '{run_id}' predates provenance stamping ({unstamped}): it "
                         "cannot be shown to match this code, so it is not a valid control")
        out["violations"] = ["no_provenance"]
        out["registry"] = {"numerics": None}
        out["current"] = {"numerics": cur_num}
        for m, e in envs.items():
            out["drift"] += [f"[{m}] {d}" for d
                             in provenance.diff_defaults(e.get("model_defaults"), cur_defaults)]
        return out

    if req.get("numerics_fingerprint", True):
        # File-by-file where the stamp carries a manifest, so a change in the SET's own definition
        # (ST-2 moved the bootstrap out of it) does not condemn a registry whose numbers are intact.
        # Scoped to the files the registry's OWN builder actually executes, derived from its import
        # graph: `pooled_runner` is value-deciding somewhere, but not here, and refusing a registry
        # over a file its pipeline never loads is how a guard earns being switched off. A scope that
        # comes back empty or implausibly small means the walk failed — then check everything.
        scope = provenance.artifact_scope(BUILDER)
        if len(scope) < MIN_SCOPE:
            scope = None
        checks = {m: provenance.verify_numerics(e.get("numerics"), scope=scope)
                  for m, e in envs.items()}
        stale = {m: v for m, v in checks.items() if not v["ok"]}
        if stale:
            out["violations"].append("numerics_fingerprint")
            first = next(iter(stale.values()))
            out["reason"] = (f"value-deciding code changed since registry '{run_id}' was built: "
                             f"{first['reason']}")
            out["changed_files"] = sorted({f for v in stale.values() for f in v["changed"]})
            for m in stale:
                out["drift"] += [f"[{m}] {d}" for d in provenance.diff_defaults(
                    envs[m].get("model_defaults"), cur_defaults)]

    for key in ("dataset_hash", "split_hash"):
        if not req.get(key, True) or cur.get(key) is None:
            continue
        bad = {m: c.get(key) for m, c in cards.items() if c.get(key) != cur[key]}
        if bad:
            out["violations"].append(key)
            out["reason"] = (out["reason"] + "; " if out["reason"] else "") + \
                f"{key} differs from the registry ({bad} vs {cur[key][:8]})"

    if req.get("git_sha", False):
        cur_sha = provenance.code_version().get("git_sha")
        bad = {m: (e.get("code") or {}).get("git_sha") for m, e in envs.items()
               if (e.get("code") or {}).get("git_sha") != cur_sha}
        if bad:
            out["violations"].append("git_sha")
            out["reason"] = (out["reason"] + "; " if out["reason"] else "") + "git SHA differs"

    out["registry"] = {"numerics": reg_num, "dataset_hash": {m: c.get("dataset_hash") for m, c in cards.items()},
                       "split_hash": {m: c.get("split_hash") for m, c in cards.items()}}
    out["current"] = {"numerics": cur_num, **cur}
    out["ok"] = not out["violations"]
    if out["ok"]:
        out["reason"] = f"registry '{run_id}' matches this checkout ({cur_num[:8]})"
    return out
