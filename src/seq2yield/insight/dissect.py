"""Dataset dissection — the EXPLORATORY arm of the harness.

The harness is confirmatory: given a comparison it answers "is candidate > baseline on R²?". This
module is its complement. It reads the per-series baseline metrics a dataset already has (the
reproduction / AutoML substrate — series × model × train_size × R²) and SURFACES questions the
council can then chase, turning trained-model results into GENERATED hypotheses. That is how the loop
proposes new directions without a human supplying the insight (the "reproduction is the substrate for
discovery" design).

Each dissection emits `GeneratedQuestion` objects: an observation grounded in numbers, a falsifiable
hypothesis, and a *suggested next experiment* expressed as an intervention axis the existing harness
can already run — so an insight closes the loop back into a RunSpec.

The core (`dissect_metrics`) is pure over a metrics DataFrame (no data files needed). `dissect_dataset`
loads a registry's metrics.csv and, when the raw dataset is present, enriches with covariate
correlations (why are the hard series hard?).
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

# thresholds (tunable; kept explicit rather than hidden constants)
_POOR_R2 = 0.30            # even the best model is weak here -> likely an irreducible ceiling
_MIN_SERIES = 4           # below this, per-series structure isn't worth dissecting
_SPREAD_HI_Q = 0.75       # a series whose cross-model R² spread is in the top quartile is "sensitive"
_SPREAD_FLOOR = 0.15      # ...but only if the spread is also materially large (guards near-flat corpora)
_DE_SLOPE = 0.02          # ΔR² per doubling that still counts as "improving" at the largest N

# map a suggested intervention back to a harness intervention_type (for PI focus hints)
_INTERVENTION_TYPE = {
    "dissect covariates / add a mechanistic feature": "feature_representation",
    "architecture comparison on the sensitive series": "model_architecture",
    "estimate the noise ceiling / add a new feature family": "feature_representation",
    "data-efficiency sweep / cross-dataset transfer": "data_efficiency",
    "stratify by the correlated covariate": "feature_representation",
}


class GeneratedQuestion(BaseModel):
    """A question the dissection surfaced from the data — not a cell picked off a fixed grid."""
    id: str
    dataset: str
    kind: str                      # series_difficulty | model_sensitivity | model_agnostic_ceiling |
    #                                data_efficiency | difficulty_covariate
    observation: str               # what the numbers show
    hypothesis: str                # a falsifiable guess about the cause
    suggested_intervention: str    # the next experiment (maps to a harness intervention_type)
    evidence: dict = Field(default_factory=dict)
    priority: float = 0.5          # 0..1, higher = chase first

    @property
    def intervention_type(self) -> str | None:
        return _INTERVENTION_TYPE.get(self.suggested_intervention)


def _qid(dataset: str, kind: str, key: str) -> str:
    h = hashlib.sha1(f"{dataset}|{kind}|{key}".encode()).hexdigest()[:8]
    return f"q_{kind}_{h}"


def _fmt(ids) -> str:
    return ", ".join(str(int(i)) for i in ids)


def _per_series(df: pd.DataFrame, train_size: int) -> pd.DataFrame:
    """Collapse to one row per series at a train size: mean R² across models + the cross-model
    spread (max−min). Iterations are averaged first."""
    sub = df[df["train_size"] == train_size]
    g = sub.groupby(["series", "model"], as_index=False)["r2"].mean()
    agg = g.groupby("series")["r2"].agg(r2_mean="mean", r2_max="max", r2_min="min")
    agg["spread"] = agg["r2_max"] - agg["r2_min"]
    return agg.sort_values("r2_mean")


def _difficulty(df, dataset, size, k=3) -> list[GeneratedQuestion]:
    ps = _per_series(df, size)
    if len(ps) < _MIN_SERIES:
        return []
    q25, med = ps["r2_mean"].quantile(0.25), ps["r2_mean"].median()
    hardest = ps[ps["r2_mean"] <= q25].head(k)
    if hardest.empty:
        return []
    ids = list(hardest.index)
    gap = float(med - hardest["r2_mean"].mean())
    return [GeneratedQuestion(
        id=_qid(dataset, "series_difficulty", _fmt(ids)),
        dataset=dataset, kind="series_difficulty",
        observation=(f"Series [{_fmt(ids)}] are systematically hard at N={size}: mean R²="
                     f"{hardest['r2_mean'].mean():.3f} vs corpus median {med:.3f} (gap {gap:.3f})."),
        hypothesis=("A property shared by these series (sequence length regime, GC content, "
                    "expression range, or higher epistasis) caps predictability."),
        suggested_intervention="dissect covariates / add a mechanistic feature",
        evidence={"series": ids, "train_size": int(size),
                  "r2_mean_hard": round(float(hardest["r2_mean"].mean()), 4),
                  "r2_median": round(float(med), 4)},
        priority=min(1.0, 0.4 + gap))]


def _sensitivity_and_ceiling(df, dataset, size) -> list[GeneratedQuestion]:
    ps = _per_series(df, size)
    if len(ps) < _MIN_SERIES:
        return []
    out: list[GeneratedQuestion] = []
    thr = max(ps["spread"].quantile(_SPREAD_HI_Q), _SPREAD_FLOOR)   # top-quartile AND materially large
    # model-SENSITIVE: big cross-model spread AND some model already does well -> architecture matters
    sens = ps[(ps["spread"] >= thr) & (ps["r2_max"] > _POOR_R2)].sort_values("spread")
    sens = sens.tail(3)
    if not sens.empty:
        ids = list(sens.index)
        out.append(GeneratedQuestion(
            id=_qid(dataset, "model_sensitivity", _fmt(ids)),
            dataset=dataset, kind="model_sensitivity",
            observation=(f"Series [{_fmt(ids)}] show large cross-model R² spread "
                         f"(up to {sens['spread'].max():.3f}) at N={size}: some architectures "
                         "capture their signal, others miss it."),
            hypothesis=("These series carry an architecture-specific pattern (e.g. positional / motif "
                        "structure a CNN sees but a bag-of-k-mers RF cannot)."),
            suggested_intervention="architecture comparison on the sensitive series",
            evidence={"series": ids, "train_size": int(size),
                      "max_spread": round(float(sens["spread"].max()), 4)},
            priority=min(1.0, 0.5 + float(sens["spread"].max()))))
    # model-AGNOSTIC ceiling: even the best model is weak -> irreducible noise or a missing feature
    ceiling = ps[ps["r2_max"] < _POOR_R2].head(3)
    if not ceiling.empty:
        ids = list(ceiling.index)
        out.append(GeneratedQuestion(
            id=_qid(dataset, "model_agnostic_ceiling", _fmt(ids)),
            dataset=dataset, kind="model_agnostic_ceiling",
            observation=(f"Series [{_fmt(ids)}] are hard for EVERY model at N={size} "
                         f"(best R²<{_POOR_R2}, small spread): no architecture helps."),
            hypothesis=("The ceiling is irreducible here — label/replicate noise or a biophysical "
                        "signal absent from all current feature sets."),
            suggested_intervention="estimate the noise ceiling / add a new feature family",
            evidence={"series": ids, "train_size": int(size),
                      "best_r2": round(float(ceiling["r2_max"].max()), 4)},
            priority=0.7))
    return out


def _data_efficiency(df, dataset) -> list[GeneratedQuestion]:
    sizes = sorted(df["train_size"].unique())
    if len(sizes) < 2:
        return []
    lo, hi = sizes[0], sizes[-1]
    a, b = _per_series(df, lo)["r2_mean"], _per_series(df, hi)["r2_mean"]
    common = a.index.intersection(b.index)
    if len(common) < _MIN_SERIES:
        return []
    slope = (b.loc[common] - a.loc[common])            # ΔR² from smallest to largest N
    improving = slope[slope > _DE_SLOPE].sort_values(ascending=False).head(3)
    if improving.empty:
        return []
    ids = list(improving.index)
    return [GeneratedQuestion(
        id=_qid(dataset, "data_efficiency", _fmt(ids)),
        dataset=dataset, kind="data_efficiency",
        observation=(f"Series [{_fmt(ids)}] are still climbing from N={lo} to N={hi} "
                     f"(ΔR² up to {float(improving.max()):.3f}): data-limited, not model-limited."),
        hypothesis="More labelled data — or transfer from a related dataset — would still raise R².",
        suggested_intervention="data-efficiency sweep / cross-dataset transfer",
        evidence={"series": ids, "delta_r2_max": round(float(improving.max()), 4),
                  "from_n": int(lo), "to_n": int(hi)},
        priority=min(1.0, 0.4 + float(improving.max())))]


def correlate_difficulty(series_r2: pd.Series, covariates: pd.DataFrame,
                         dataset: str = "unknown", *, min_abs_r: float = 0.4) -> list[GeneratedQuestion]:
    """Pure: given per-series mean R² and a per-series covariate table (gc, length_cv, expr_mean,
    expr_std, ...), report which covariate best explains difficulty. Emits a `difficulty_covariate`
    question naming the covariate and the correlation (the 'why are the hard series hard' answer)."""
    common = series_r2.index.intersection(covariates.index)
    if len(common) < _MIN_SERIES:
        return []
    y = series_r2.loc[common].astype(float)
    best = None
    for col in covariates.columns:
        x = pd.to_numeric(covariates.loc[common, col], errors="coerce").astype(float)
        if x.notna().sum() < _MIN_SERIES or x.std() == 0:
            continue
        r = float(np.corrcoef(x.fillna(x.mean()), y)[0, 1])
        if best is None or abs(r) > abs(best[1]):
            best = (col, r)
    if not best or abs(best[1]) < min_abs_r:
        return []
    col, r = best
    direction = "lower" if r > 0 else "higher"
    return [GeneratedQuestion(
        id=_qid(dataset, "difficulty_covariate", col),
        dataset=dataset, kind="difficulty_covariate",
        observation=(f"Per-series R² correlates with '{col}' (r={r:+.2f}): series with {direction} "
                     f"'{col}' are the hard ones."),
        hypothesis=f"'{col}' drives predictability; encoding it explicitly should recover the hard series.",
        suggested_intervention="stratify by the correlated covariate",
        evidence={"covariate": col, "pearson_r": round(r, 3), "n_series": int(len(common))},
        priority=min(1.0, 0.5 + abs(r) / 2))]


def dissect_metrics(df: pd.DataFrame, dataset: str = "unknown",
                    train_size: int | None = None) -> list[GeneratedQuestion]:
    """Pure dissection over a per-series metrics frame (columns: series, model, train_size, r2).
    Returns generated questions sorted by priority (highest first)."""
    if df.empty or "series" not in df.columns:
        return []
    size = train_size if train_size is not None else int(max(df["train_size"]))
    qs = _difficulty(df, dataset, size) + _sensitivity_and_ceiling(df, dataset, size) \
        + _data_efficiency(df, dataset)
    return sorted(qs, key=lambda q: q.priority, reverse=True)


def _series_covariates(dataset: str):
    """Per-series covariates (gc mean, length CV, expression mean/std) from the raw dataset, or None
    when the data isn't present. Best-effort: any failure returns None so the pure dissections still
    run. (The covariate loader is intentionally defensive — the harness never depends on it.)"""
    try:
        from ..data import datasets
        if not datasets.data_present(dataset):
            return None
        from ..data.cleaning import SEQ_COL, TARGET_COL
        from ..experiments import pooled_runner
        frame = pooled_runner._frame(dataset)
        if "series" not in frame.columns:
            return None
        rows = {}
        for sid, g in frame.groupby("series"):
            seqs = g[SEQ_COL].astype(str)
            gc = seqs.apply(lambda s: (s.count("G") + s.count("C")) / max(1, len(s))).mean()
            lens = seqs.str.len()
            y = pd.to_numeric(g[TARGET_COL], errors="coerce")
            rows[sid] = {"gc": float(gc),
                         "length_cv": float(lens.std() / lens.mean()) if lens.mean() else 0.0,
                         "expr_mean": float(y.mean()), "expr_std": float(y.std())}
        return pd.DataFrame.from_dict(rows, orient="index") if rows else None
    except Exception:
        return None


def dissect_dataset(dataset: str, metrics_csv, train_size: int | None = None) -> list[GeneratedQuestion]:
    """Load a registry metrics.csv for `dataset`, run the pure dissections, and (when the raw data is
    present) add the covariate-correlation question. `metrics_csv` is a path to the baseline run's
    metrics.csv (e.g. experiments/runs/<baseline_run_id>/metrics.csv)."""
    df = pd.read_csv(metrics_csv)
    qs = dissect_metrics(df, dataset, train_size)
    cov = _series_covariates(dataset)
    if cov is not None:
        size = train_size if train_size is not None else int(max(df["train_size"]))
        qs += correlate_difficulty(_per_series(df, size)["r2_mean"], cov, dataset)
    return sorted(qs, key=lambda q: q.priority, reverse=True)


def to_focus_hints(questions: list[GeneratedQuestion]) -> list[str]:
    """Distinct harness intervention_types the questions point at — a data-driven prior the PI can
    fold into its focus (so exploration follows the data, not just the uncovered grid)."""
    seen, out = set(), []
    for q in questions:
        it = q.intervention_type
        if it and it not in seen:
            seen.add(it)
            out.append(it)
    return out
