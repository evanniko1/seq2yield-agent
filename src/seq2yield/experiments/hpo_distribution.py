"""C5 — the per-series / per-subregion HPO-DISTRIBUTION study (the Nat Comms question).

The original E. coli benchmark (56 mutational series) never asked: does every series prefer the SAME
architecture, or do the best hyperparameters vary from series to series? This runs the C2 hybrid
search — UNDER the C10 gate, bounded + async so it never hangs — independently on each unit (each
E. coli series, or each level of a C6 stratum on a pooled dataset), collects the argmax config per
unit, and reports the DISTRIBUTION of the best {kernel_size, lr, dropout, …} across units per model
class. A wide distribution ⇒ per-unit heterogeneity (no universal optimum); a tight one ⇒ a shared
sweet spot.

Depends on C1 (the space), C2 (the search), C10 (the gate), C6 (strata units).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..data import datasets
from ..models import registry as reg

ROOT = Path(__file__).resolve().parents[3]

# the headline knobs to highlight per model class (the report includes every flattened knob too)
HEADLINE = {
    "cnn": ["kernel_sizes_0", "kernel_sizes_mean", "lr", "dropout"],
    "transformer": ["d_model", "layers", "lr", "dropout"],
    "rf": ["n_estimators", "max_depth", "max_features"],
    "mlp": ["hidden_layer_sizes_0", "alpha", "learning_rate_init"],
}


@dataclass
class UnitResult:
    unit: str
    best_config: dict
    best_r2: float
    gate_action: str
    n_evals: int
    timed_out: bool = False


@dataclass
class HPODistribution:
    dataset: str
    model: str
    unit_type: str                 # 'series' | '<stratum>'
    train_size: int
    strategy: str
    units: list[str]
    per_unit: list[UnitResult] = field(default_factory=list)
    distribution: dict = field(default_factory=dict)     # feature -> summary
    heterogeneous: dict = field(default_factory=dict)    # feature -> bool
    headline: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        d = {k: v for k, v in asdict(self).items() if k != "per_unit"}
        d["per_unit"] = [asdict(u) for u in self.per_unit]
        return d


# ------------------------------------------------------------------ config flattening + summary
def _flatten(config: dict) -> dict:
    """Turn a config into scalar features: a list knob → its first element, mean, and depth."""
    out = {}
    for k, v in (config or {}).items():
        if isinstance(v, (list, tuple)):
            if v:
                out[f"{k}_0"] = v[0]
                out[f"{k}_mean"] = float(np.mean(v))
                out[f"{k}_depth"] = len(v)
        else:
            out[k] = v
    return out


def _summarize(values: list) -> tuple[dict, bool]:
    """Numeric → mean/std/min/max/cv (+ heterogeneous if CV≥0.15); categorical → counts/mode
    (+ heterogeneous if >1 distinct value)."""
    numeric = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if len(numeric) == len(values) and values:
        arr = np.asarray(numeric, dtype=float)
        mean = float(arr.mean())
        std = float(arr.std())
        cv = float(std / abs(mean)) if mean else (0.0 if std == 0 else float("inf"))
        return ({"kind": "numeric", "n": len(arr), "mean": round(mean, 4), "std": round(std, 4),
                 "min": round(float(arr.min()), 4), "max": round(float(arr.max()), 4),
                 "cv": round(cv, 3), "values": [round(float(x), 4) for x in arr]},
                bool(cv >= 0.15))
    counts: dict = {}
    for v in values:
        counts[str(v)] = counts.get(str(v), 0) + 1
    mode = max(counts, key=counts.get) if counts else None
    return ({"kind": "categorical", "n": len(values), "counts": counts, "mode": mode,
             "n_distinct": len(counts), "values": [str(v) for v in values]},
            bool(len(counts) > 1))


# ------------------------------------------------------------------ unit enumeration
def _units(dataset: str, unit_type: str, n_units: int) -> list[str]:
    ds = datasets.spec(dataset)
    if unit_type == "series":
        if ds.structure != "per_series":
            raise ValueError(f"'series' units require a per_series dataset; {dataset} is {ds.structure}")
        from ..data.loaders import load_split_csv
        from ..data.splits import load_manifest
        man = load_manifest(ROOT / "data/splits")
        it = next(iter(man["iterations"]))
        work = load_split_csv(man["iterations"][it]["working_set"]["path"])
        return [str(s) for s in sorted(work["mut_series"].unique())[:n_units]]
    # else: a stratum name on a pooled dataset -> one unit per level
    from ..data import strata
    if unit_type not in strata.applicable(dataset):
        raise ValueError(f"stratum '{unit_type}' not available for {dataset} "
                         f"(have {strata.applicable(dataset)})")
    return [f"{unit_type}={lvl}" for lvl in strata.levels(dataset, unit_type)]


# ------------------------------------------------------------------ the study
def run_hpo_distribution(dataset: str, model: str = "cnn", *, unit_type: str = "series",
                         n_units: int = 8, train_size: int = 800, strategy: str = "bandit",
                         min_action: str = "light", feature_set: str = "one_hot",
                         feature_scaling: str = "auto", seed: int = 0,
                         gate_kwargs: dict | None = None) -> HPODistribution:
    """Search each unit under the C10 gate and report the distribution of the winning knobs.

    `unit_type='series'` iterates E. coli mutational series; otherwise it is a C6 stratum name
    (e.g. 'gc_bin') and units are its levels on a pooled dataset. `min_action` forces at least a
    light search per unit (a study wants a result everywhere), still bounded/async via the gate."""
    from agents import biology_architect, search_gate
    units = _units(dataset, unit_type, n_units)
    per_unit: list[UnitResult] = []
    region = biology_architect.search_region(dataset, model)
    seeds = biology_architect.seed_configs(dataset, model)

    for u in units:
        ctx = search_gate.build_context(model, dataset, subregion=u, memory_records=[])
        out = search_gate.run_gated(ctx, seeds=seeds, space=region, feature_set=feature_set,
                                    feature_scaling=feature_scaling, seed=seed,
                                    min_action=min_action, log=True, **(gate_kwargs or {}))
        cfg = dict(out.result.best_config) if out.result else {}
        per_unit.append(UnitResult(
            unit=u, best_config=cfg,
            best_r2=(round(float(out.result.best_score), 4) if out.result else float("nan")),
            gate_action=out.decision.action,
            n_evals=(out.result.n_evals if out.result else 0),
            timed_out=out.timed_out))

    # aggregate the distribution across units
    flat = [_flatten(u.best_config) for u in per_unit if u.best_config]
    feats = sorted({k for f in flat for k in f})
    distribution, heterogeneous = {}, {}
    for feat in feats:
        vals = [f[feat] for f in flat if feat in f]
        if len(vals) < 2:
            continue
        summary, het = _summarize(vals)
        distribution[feat] = summary
        heterogeneous[feat] = het

    return HPODistribution(
        dataset=dataset, model=model, unit_type=unit_type, train_size=train_size, strategy=strategy,
        units=units, per_unit=per_unit, distribution=distribution, heterogeneous=heterogeneous,
        headline=[h for h in HEADLINE.get(model, []) if h in distribution])


def record_study(res: HPODistribution, out_dir: str | Path | None = None) -> Path:
    """Persist the study to reports/hpo_distributions/<dataset>-<model>-<unit_type>-<ts>.json."""
    import json
    d = Path(out_dir or ROOT / "reports" / "hpo_distributions")
    d.mkdir(parents=True, exist_ok=True)
    ts = f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
    p = d / f"{res.dataset}-{res.model}-{res.unit_type}-{ts}.json"
    p.write_text(json.dumps(res.as_dict(), indent=2, default=str), encoding="utf-8")
    return p
