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
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[3]
# baseline registry metrics per dataset (per-series R²). ecoli's full 56-series registry is the
# canonical one; other datasets resolve as they are onboarded (missing -> no hints, gracefully).
_DEFAULT_METRICS = {"ecoli": ROOT / "experiments/runs/2026-06-04-full56/metrics.csv"}

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


def _covariate_frame(dataset: str):
    """Raw (series, Sequence, Protein) frame keyed by SERIES_COL, for BOTH structures:
    per-series datasets (E. coli) read the split working set; pooled datasets use the pooled frame."""
    from ..data import datasets
    from ..data.cleaning import SEQ_COL, SERIES_COL, TARGET_COL
    ds = datasets.spec(dataset) if datasets.exists(dataset) else None
    structure = ds.structure if ds else ("pooled" if dataset == "yeast" else "per_series")
    if structure == "per_series":                     # E. coli: all series' training rows
        from ..data.loaders import load_split_csv
        from ..data.splits import load_manifest
        m = load_manifest(ROOT / "data/splits")
        it = list(m["iterations"])[0]
        frame = load_split_csv(m["iterations"][it]["working_set"]["path"])
    else:
        from ..experiments import pooled_runner
        frame = pooled_runner._frame(dataset)
    cols = [c for c in (SERIES_COL, SEQ_COL, TARGET_COL) if c in frame.columns]
    return frame[cols] if SERIES_COL in cols else None


def _series_covariates(dataset: str):
    """Per-series covariates (gc mean, length CV, expression mean/std) keyed by the SAME series id as
    the baseline metrics — so `correlate_difficulty` can answer 'why are the hard series hard?'.
    Best-effort: any failure returns None so the pure dissections still run (the harness never
    depends on it). Works for E. coli (natural mutational series) and pooled datasets with groups."""
    try:
        from ..data import datasets
        if not datasets.data_present(dataset):
            return None
        from ..data.cleaning import SEQ_COL, SERIES_COL, TARGET_COL
        frame = _covariate_frame(dataset)
        if frame is None or SERIES_COL not in frame.columns:
            return None
        rows = {}
        for sid, g in frame.groupby(SERIES_COL):
            seqs = g[SEQ_COL].astype(str)
            gc = seqs.apply(lambda s: (s.count("G") + s.count("C")) / max(1, len(s))).mean()
            lens = seqs.str.len()
            y = pd.to_numeric(g[TARGET_COL], errors="coerce")
            key = int(sid) if str(sid).lstrip("-").isdigit() else sid
            rows[key] = {"gc": float(gc),
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


# --------------------------------------------------------------------- PI wiring ---
def default_metrics_path(dataset: str):
    """Path to a dataset's baseline metrics.csv, or None if it hasn't been reproduced yet."""
    p = _DEFAULT_METRICS.get(dataset)
    return p if p and p.exists() else None


def _insight_path(dataset: str) -> Path:
    return ROOT / "experiments" / "insights" / f"{dataset}.jsonl"


def save_questions(dataset: str, questions: list[GeneratedQuestion]) -> Path:
    p = _insight_path(dataset)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(q.model_dump_json() for q in questions), encoding="utf-8")
    return p


def load_questions(dataset: str) -> list[GeneratedQuestion]:
    """Persisted generated questions for a dataset (written by scripts/run_insight.py), or []."""
    p = _insight_path(dataset)
    if not p.exists():
        return []
    return [GeneratedQuestion.model_validate_json(ln)
            for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


def pooled_predictions(dataset: str, *, model_family: str = "rf", feature_set: str = "one_hot",
                       seed: int = 1, frac: float = 0.1, max_train: int = 6000):
    """A pooled baseline's held-out predictions via the HARNESS pooled path (not a shadow pipeline):
    returns (test_sequences, y_true, y_pred), positionally aligned. Deterministic given the seed;
    train rows capped for speed. This is the real prediction source neighborhood dissection needs."""
    from types import SimpleNamespace

    from ..data.cleaning import SEQ_COL, TARGET_COL
    from ..experiments import pooled_runner as P
    spec = SimpleNamespace(dataset=dataset, seed=seed, feature_set=feature_set,
                           feature_scaling="none", hyperparameters={}, sampling_policy="random")
    train_full, test = P.holdout(spec, frac=frac)
    train = P.subsample(train_full, min(max_train, len(train_full)), "random", seed)
    y_pred = P.fit_predict(spec, model_family, train, test)
    return test[SEQ_COL].tolist(), test[TARGET_COL].to_numpy(), y_pred


def dissect_any(dataset: str, *, k: int = 8, train_size: int | None = None) -> list[GeneratedQuestion]:
    """Structure-aware dissection. Per-series datasets (E. coli) dissect their baseline registry
    metrics; pooled datasets (yeast, dream2022, ...) DISCOVER neighborhoods from a real pooled
    baseline's held-out predictions and dissect those. [] when neither is available."""
    path = default_metrics_path(dataset)
    if path:
        return dissect_dataset(dataset, path, train_size=train_size)
    try:
        from ..data import datasets as ds
        if ds.exists(dataset) and ds.spec(dataset).structure == "pooled" and ds.data_present(dataset):
            seqs, y_true, y_pred = pooled_predictions(dataset)
            return dissect_pooled_predictions(seqs, y_true, y_pred, dataset, k=k)
    except Exception:
        pass
    return []


def hints_for_dataset(dataset: str, metrics_csv=None) -> tuple[list[str], list[GeneratedQuestion]]:
    """(focus_hints, questions) for one dataset — the CHEAP path used inside the council loop:
    per-series datasets dissect their metrics.csv live (no training); pooled datasets read the
    persisted insight file (computed once by scripts/run_insight.py, which does the pooled fit).
    Graceful ([], []) when neither exists."""
    path = metrics_csv or default_metrics_path(dataset)
    if path and Path(path).exists():
        qs = dissect_dataset(dataset, path)
    else:
        qs = load_questions(dataset)                 # persisted pooled-neighborhood questions
    return to_focus_hints(qs), qs


def aggregate_focus_hints(datasets=None) -> tuple[list[str], dict]:
    """Union of data-driven focus hints across ready datasets (order preserved), plus each dataset's
    questions. Graceful: datasets without baselines contribute nothing. This is what the PI folds in
    so exploration follows the observed structure across the whole corpus."""
    if datasets is None:
        try:
            from ..data import datasets as ds_mod
            datasets = ds_mod.ready_ids()
        except Exception:
            datasets = list(_DEFAULT_METRICS)
    hints: list[str] = []
    per: dict = {}
    for d in datasets:
        h, qs = hints_for_dataset(d)
        per[d] = qs
        for x in h:
            if x not in hints:
                hints.append(x)
    return hints, per


# ------------------------------------------ neighborhood discovery (pooled datasets) ---
# E. coli is naturally partitioned into mutational series; pooled datasets (yeast, dream2022, ...)
# are not. To dissect them per-neighborhood we DISCOVER clusters in sequence space first, treat each
# cluster as a series-like unit, then dissect exactly as for E. coli. Cluster-level exploration comes
# first; full-dataset (global) exploration follows once the per-neighborhood results are in.
def _kmer_matrix(sequences, k: int = 3) -> np.ndarray:
    from itertools import product
    idx = {"".join(p): i for i, p in enumerate(product("ACGT", repeat=k))}
    X = np.zeros((len(sequences), len(idx)), dtype=float)
    for r, s in enumerate(sequences):
        s = str(s).upper()
        for i in range(len(s) - k + 1):
            j = idx.get(s[i:i + k])
            if j is not None:
                X[r, j] += 1
    row = X.sum(1, keepdims=True)
    row[row == 0] = 1.0
    return X / row                          # length-normalized k-mer composition


def cluster_sequences(sequences, k: int = 8, *, seed: int = 0, kmer: int = 3) -> np.ndarray:
    """Discover k neighborhoods in sequence space (length-normalized k-mer composition -> KMeans).
    Deterministic for a given seed. Clamps k to the sample size; k<=1 -> a single neighborhood."""
    n = len(sequences)
    if n == 0:
        return np.array([], dtype=int)
    k = max(1, min(k, n))
    if k == 1:
        return np.zeros(n, dtype=int)
    from sklearn.cluster import KMeans
    X = _kmer_matrix(sequences, kmer)
    return KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(X).astype(int)


def neighborhoods_to_metrics_frame(labels, y_true, y_pred, *, model: str = "pooled",
                                   train_size: int = 0, min_n: int = 20) -> pd.DataFrame:
    """Per-neighborhood R² from a pooled model's held-out predictions -> a dissect-compatible frame
    (series = neighborhood id). Neighborhoods with < min_n test points are dropped (unstable R²)."""
    from ..statistics.bootstrap import _r2
    y_true, y_pred, labels = np.asarray(y_true, float), np.asarray(y_pred, float), np.asarray(labels)
    rows = []
    for c in np.unique(labels):
        m = labels == c
        if int(m.sum()) < min_n:
            continue
        rows.append({"iteration": "iteration_1", "series": int(c), "model": model,
                     "train_size": int(train_size), "r2": _r2(y_true[m], y_pred[m]),
                     "n": int(m.sum())})
    return pd.DataFrame(rows)


def dissect_pooled_predictions(sequences, y_true, y_pred, dataset: str = "unknown", *,
                               k: int = 8, seed: int = 0, min_n: int = 20) -> list[GeneratedQuestion]:
    """Discover neighborhoods in a pooled dataset and dissect them: cluster the held-out sequences,
    compute per-neighborhood R² from an existing pooled model's predictions, and generate questions
    (difficulty / ceiling; sensitivity needs multiple models). Pure over the arrays it is given —
    the loop calls it once a pooled baseline has produced test predictions."""
    labels = cluster_sequences(sequences, k=k, seed=seed)
    frame = neighborhoods_to_metrics_frame(labels, y_true, y_pred,
                                           train_size=len(y_true), min_n=min_n)
    qs = dissect_metrics(frame, dataset)
    for q in qs:                            # mark that the unit was discovered, not a natural series
        q.evidence["unit"] = "discovered_neighborhood"
    return qs
