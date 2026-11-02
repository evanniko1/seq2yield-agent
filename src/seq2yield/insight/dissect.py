"""Dataset dissection — the EXPLORATORY arm of the harness.

The harness is confirmatory: given a comparison it answers "is candidate > baseline on R²?". This
module is its complement. It reads the per-unit baseline metrics a dataset already has (the
reproduction / AutoML substrate — unit × model × train_size × metric) and SURFACES questions the
council can then chase, turning trained-model results into GENERATED hypotheses. That is how the
loop proposes new directions without a human supplying the insight.

Each dissection emits `GeneratedQuestion` objects: an observation grounded in numbers, a falsifiable
hypothesis, and a *suggested next experiment* expressed as an intervention axis the existing harness
can already run — so an insight closes the loop back into a RunSpec.

REBUILT 2026-11-02 (audit SA-6 / ST-8 / ML-10). The previous version's "why" was built out of the
metric it was explaining:

* difficulty was per-unit R², whose denominator is the unit's own Var(y), and the covariate family
  included `expr_mean` / `expr_std` — the denominator itself. On E. coli this produced
  "lower expr_std -> hard (r=+0.53)" while per-series RMSE correlated POSITIVELY with expr_std
  (+0.44): the low-variance series had SMALLER errors and low R² only through a small Var(y).
* the reported covariate was the max |r| over the family with no p-value, which under H0 reports
  something 69% of the time at the 8 units a pooled dissection actually has.
* per-cluster R² ranked clusters by their own Var(y): on yeast the rank correlation between
  per-cluster R² and per-cluster MSE is +0.04, and the lowest-MSE cluster of seven was named as
  one of the three "hard for EVERY model" — from a SINGLE model family.

The invariant this module now holds: **the observed target `y` enters an explanation only through
an out-of-fold prediction.** Difficulty is an error scale (RMSE, or RMSE over a replicate-derived
noise ceiling where a dataset declares one); the covariate family is sequence- and design-derived
only; every covariate report carries a family-wise max-T permutation p-value
(`seq2yield.statistics.selection`); clusters are fitted on TRAIN sequences and ranked against a
y-matched random partition rather than against their own variance; and "hard for every model"
requires at least two model families.

Target-moment covariates are rejected by name (`_is_target_derived`), so passing an old covariate
frame cannot reintroduce SA-6. Prediction-MOMENT covariates (the SD of the predictions in a unit)
are deliberately NOT in the default family either: they are a proxy for the same scale and would
re-tell the SA-6 story with an out-of-fold label on it.

The core (`dissect_metrics`) is pure over a metrics DataFrame (no data files needed).
`dissect_dataset` loads a registry's metrics.csv and, when the raw dataset is present, enriches with
covariate correlations (why are the hard units hard?).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from ..statistics import selection

ROOT = Path(__file__).resolve().parents[3]


# baseline registry metrics per dataset (per-series metrics). ecoli's full 56-series registry is the
# canonical one; other datasets resolve as they are onboarded (missing -> no hints, gracefully).
def _registry_metrics(dataset: str):
    """The dataset's baseline per-series metrics. ecoli resolves to the CANONICAL registry from
    configs/baseline_registry.yaml rather than a hardcoded dated folder, so repointing the registry
    repoints the insight layer with it (ML-02); other datasets resolve as they are onboarded."""
    if dataset != "ecoli":
        return None
    try:
        from orchestration.baseline_guard import registry_path, registry_ready
        # readiness, not existence: the builder checkpoints metrics.csv as it goes, so a registry
        # mid-build would otherwise be dissected as if it were the whole corpus.
        return registry_path("metrics.csv") if registry_ready() else None
    except Exception:
        return None


_DEFAULT_DATASETS = ("ecoli",)
# phase.py's fallback referenced this name; kept as an alias so that path cannot NameError.
_DEFAULT_METRICS = _DEFAULT_DATASETS

# thresholds (tunable; kept explicit rather than hidden constants)
_POOR_R2 = 0.30            # legacy r2_raw scale only: best model this weak -> likely a ceiling
_MIN_SERIES = 4           # below this, per-unit structure isn't worth dissecting
_SPREAD_HI_Q = 0.75       # a unit whose cross-model spread is in the top quartile is "sensitive"
_SPREAD_FLOOR = 0.15      # ...but only if the spread is also materially large (near-flat corpora)
_DE_SLOPE = 0.02          # Δdifficulty per sweep that still counts as "improving" at the largest N
_HARD_Q = 0.75            # a unit in the top quartile of the difficulty scale is "hard"
_ALPHA = 0.05             # family-wise level for BOTH the covariate test and the cluster null

# Difficulty scales, in preference order. The emitted question always NAMES its scale, because
# "hard" means something different on each of them (audit SA-6).
SCALE_NOISE = "rmse_over_noise_ceiling"   # RMSE / replicate-derived noise RMSE: 1.0 = at the ceiling
SCALE_RMSE = "rmse"                       # RMSE in the target's own units; no Var(y) in it
SCALE_R2_RAW = "r2_raw"                   # 1 - MSE/Var(y): Var(y) IS the denominator (SA-6)
_SCALE_NOTE = {
    SCALE_NOISE: "RMSE over the replicate noise ceiling; 1.0 = at the ceiling",
    SCALE_RMSE: "RMSE, target units; no Var(y) in it",
    SCALE_R2_RAW: ("raw R² = 1 - MSE/Var(y); the unit's own target variance is the denominator, so "
                   "a low value can mean small Var(y) rather than large error — audit SA-6"),
}

# map a suggested intervention back to a harness intervention_type (for PI focus hints)
_INTERVENTION_TYPE = {
    "dissect covariates / add a mechanistic feature": "feature_representation",
    "architecture comparison on the sensitive series": "model_architecture",
    "estimate the noise ceiling / add a new feature family": "feature_representation",
    "data-efficiency sweep / cross-dataset transfer": "data_efficiency",
    "stratify by the correlated covariate": "feature_representation",
    "second model family on the weak units": "model_architecture",
}

# Covariates that are functions of the OBSERVED target. Rejecting them by name is what makes SA-6
# un-reintroducible: an old covariate frame can still be passed in, but its target moments are
# dropped with a recorded reason instead of winning the selection.
_TARGET_DERIVED_EXACT = {
    "expr_mean", "expr_std", "expr_var", "expr_range", "expr_cv", "expr_min", "expr_max",
    "y_mean", "y_std", "y_var", "target_mean", "target_std", "target_var",
    "protein_mean", "protein_std", "mean_expression", "expression_std",
}
_TARGET_DERIVED_TOKENS = ("expr", "target_", "_target", "expression", "protein_mean", "protein_std",
                          "y_mean", "y_std", "y_var", "label_mean", "label_std")


def _is_target_derived(name: str) -> bool:
    """Is this covariate name a function of the observed target? Conservative: a name that LOOKS
    like a target moment is rejected, because a false reject costs one covariate and a false accept
    costs a scientific claim (audit SA-6)."""
    n = str(name).strip().lower()
    if n in _TARGET_DERIVED_EXACT:
        return True
    return any(tok in n for tok in _TARGET_DERIVED_TOKENS)


class GeneratedQuestion(BaseModel):
    """A question the dissection surfaced from the data — not a cell picked off a fixed grid."""
    id: str
    dataset: str
    kind: str                      # series_difficulty | model_sensitivity | model_agnostic_ceiling |
    #                                single_model_weak | data_efficiency | difficulty_covariate
    observation: str               # what the numbers show
    hypothesis: str                # a falsifiable guess about the cause
    suggested_intervention: str    # the next experiment (maps to a harness intervention_type)
    evidence: dict = Field(default_factory=dict)
    priority: float = 0.5          # 0..1, higher = chase first
    # The scale "hard" was measured on. The default is what a question PERSISTED BEFORE the
    # 2026-11-02 rebuild was actually computed on, so old experiments/insights/*.jsonl lines
    # deserialize honestly instead of inheriting the new scale's name.
    difficulty_scale: str = "r2_raw_legacy"

    @property
    def intervention_type(self) -> str | None:
        return _INTERVENTION_TYPE.get(self.suggested_intervention)

    @property
    def pre_rebuild(self) -> bool:
        """True for a question loaded from a file written before the SA-6/ST-8/ML-10 rebuild: its
        difficulty is raw R² and its covariate (if any) carries no family-wise p-value."""
        return self.difficulty_scale == "r2_raw_legacy"


def _qid(dataset: str, kind: str, key: str) -> str:
    h = hashlib.sha1(f"{dataset}|{kind}|{key}".encode()).hexdigest()[:8]
    return f"q_{kind}_{h}"


def _fmt(ids) -> str:
    return ", ".join(str(int(i)) for i in ids)


# ------------------------------------------------------------------- the difficulty scale ---
def replicate_noise_rmse(frame: pd.DataFrame, replicate_cols, unit_col) -> dict:
    """Per-unit noise-ceiling RMSE from technical replicates: the pooled within-row SD across the
    replicate columns, which is the RMSE a perfect model would still pay.

    Used to put difficulty on `SCALE_NOISE`. No DatasetSpec in this corpus declares
    `replicate_cols` (checked 2026-11-02 across all seven specs), so this path is implemented and
    tested but inert here — which is why the shipped default scale is plain RMSE and not a
    noise-normalised R².
    """
    cols = [c for c in (replicate_cols or []) if c in frame.columns]
    if len(cols) < 2 or unit_col not in frame.columns:
        return {}
    out = {}
    for unit, g in frame.groupby(unit_col):
        reps = g[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        # within-row variance across replicates, averaged over rows -> the irreducible MSE
        var = np.nanvar(reps, axis=1, ddof=1)
        var = var[np.isfinite(var)]
        if var.size:
            key = int(unit) if str(unit).lstrip("-").isdigit() else unit
            out[key] = float(np.sqrt(np.mean(var)))
    return out


def _per_unit(df: pd.DataFrame, train_size: int) -> pd.DataFrame:
    """Collapse to one row per unit at a train size. Iterations are averaged first, then models.

    Adds a single `difficulty` column on which HIGHER ALWAYS MEANS HARDER, plus `difficulty_best`
    (the best model family's difficulty) and `difficulty_spread` (across families). The scale is
    recorded on the frame as `frame.attrs["difficulty_scale"]`.
    """
    sub = df[df["train_size"] == train_size]
    have = [c for c in ("r2", "rmse", "rmse_noise") if c in sub.columns]
    g = sub.groupby(["series", "model"], as_index=False)[have].mean()
    agg = g.groupby("series")[have].agg(["mean", "max", "min"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg["n_model_families"] = g.groupby("series")["model"].nunique()
    if "n" in sub.columns:
        agg["n_rows"] = sub.groupby("series")["n"].max()
    # legacy columns some readers still expect
    if "r2_mean" in agg:
        agg["spread"] = agg["r2_max"] - agg["r2_min"]

    if "rmse_mean" in agg and "rmse_noise_mean" in agg and (agg["rmse_noise_mean"] > 0).all():
        scale = SCALE_NOISE
        agg["difficulty"] = agg["rmse_mean"] / agg["rmse_noise_mean"]
        agg["difficulty_best"] = agg["rmse_min"] / agg["rmse_noise_mean"]
        agg["difficulty_worst"] = agg["rmse_max"] / agg["rmse_noise_mean"]
    elif "rmse_mean" in agg:
        scale = SCALE_RMSE
        agg["difficulty"] = agg["rmse_mean"]
        agg["difficulty_best"] = agg["rmse_min"]
        agg["difficulty_worst"] = agg["rmse_max"]
    else:
        scale = SCALE_R2_RAW                     # no error column available -> negated raw R²
        agg["difficulty"] = -agg["r2_mean"]
        agg["difficulty_best"] = -agg["r2_max"]
        agg["difficulty_worst"] = -agg["r2_min"]
    agg["difficulty_spread"] = agg["difficulty_worst"] - agg["difficulty_best"]
    agg.attrs["difficulty_scale"] = scale
    return agg.sort_values("difficulty", ascending=False)         # hardest first


def _per_series(df: pd.DataFrame, train_size: int) -> pd.DataFrame:
    """Back-compatible view: one row per series with r2_mean/r2_max/r2_min/spread, sorted by
    r2_mean ascending. Kept for readers that predate the difficulty-scale rebuild."""
    agg = _per_unit(df, train_size)
    cols = [c for c in ("r2_mean", "r2_max", "r2_min", "spread") if c in agg.columns]
    return agg[cols].sort_values(cols[0]) if cols else agg


def _scale_of(ps: pd.DataFrame) -> str:
    return ps.attrs.get("difficulty_scale", SCALE_R2_RAW)


def _d(ps: pd.DataFrame, unit, col="difficulty") -> float:
    return float(ps.loc[unit, col])


def _scale_phrase(scale: str) -> str:
    return f"scale: {scale} — {_SCALE_NOTE.get(scale, 'unspecified')}"


def _difficulty(df, dataset, size, k=3) -> list[GeneratedQuestion]:
    ps = _per_unit(df, size)
    if len(ps) < _MIN_SERIES:
        return []
    scale = _scale_of(ps)
    q_hi, med = ps["difficulty"].quantile(_HARD_Q), ps["difficulty"].median()
    hardest = ps[ps["difficulty"] >= q_hi].head(k)
    if hardest.empty:
        return []
    ids = list(hardest.index)
    gap = float(hardest["difficulty"].mean() - med)
    ev = {"series": ids, "train_size": int(size), "difficulty_scale": scale,
          "difficulty_hard_mean": round(float(hardest["difficulty"].mean()), 4),
          "difficulty_median": round(float(med), 4),
          "difficulty_gap": round(gap, 4)}
    if scale == SCALE_R2_RAW:                 # keep the legacy keys readable on the legacy scale
        ev["r2_mean_hard"] = round(float(hardest["r2_mean"].mean()), 4)
        ev["r2_median"] = round(float(ps["r2_mean"].median()), 4)
    # a gap is only comparable to 1.0 on the R² scale; normalise it for the priority on error scales
    norm_gap = gap if scale == SCALE_R2_RAW else (gap / abs(med) if med else gap)
    return [GeneratedQuestion(
        id=_qid(dataset, "series_difficulty", _fmt(ids)),
        dataset=dataset, kind="series_difficulty", difficulty_scale=scale,
        observation=(f"Units [{_fmt(ids)}] are the hardest at N={size} ({_scale_phrase(scale)}): "
                     f"difficulty {hardest['difficulty'].mean():.3f} vs corpus median {med:.3f} "
                     f"(gap {gap:+.3f})."),
        hypothesis=("A property shared by these units (sequence composition, within-unit sequence "
                    "diversity, distance from the training distribution, or unit size) caps "
                    "predictability."),
        suggested_intervention="dissect covariates / add a mechanistic feature",
        evidence=ev,
        priority=min(1.0, 0.4 + min(0.6, abs(norm_gap))))]


def _sensitivity_and_ceiling(df, dataset, size) -> list[GeneratedQuestion]:
    ps = _per_unit(df, size)
    if len(ps) < _MIN_SERIES:
        return []
    scale = _scale_of(ps)
    out: list[GeneratedQuestion] = []
    n_fam = int(ps["n_model_families"].max())
    spread_col = "spread" if scale == SCALE_R2_RAW and "spread" in ps.columns else "difficulty_spread"
    thr = max(ps[spread_col].quantile(_SPREAD_HI_Q), _SPREAD_FLOOR)   # top-quartile AND material
    hard_q, med = ps["difficulty"].quantile(_HARD_Q), ps["difficulty"].median()

    # model-SENSITIVE: big cross-family spread AND some family already does well -> architecture
    # matters. Meaningless with one family, so it is gated on >= 2 explicitly.
    if n_fam >= 2:
        sens = ps[(ps[spread_col] >= thr) & (ps["difficulty_best"] <= med)].sort_values(spread_col)
        sens = sens.tail(3)
        if not sens.empty:
            ids = list(sens.index)
            out.append(GeneratedQuestion(
                id=_qid(dataset, "model_sensitivity", _fmt(ids)),
                dataset=dataset, kind="model_sensitivity", difficulty_scale=scale,
                observation=(f"Units [{_fmt(ids)}] show large cross-model spread "
                             f"(up to {sens[spread_col].max():.3f}, {_scale_phrase(scale)}) at "
                             f"N={size} over {n_fam} model families: some architectures capture "
                             "their signal, others miss it."),
                hypothesis=("These units carry an architecture-specific pattern (e.g. positional / "
                            "motif structure a CNN sees but a bag-of-k-mers RF cannot)."),
                suggested_intervention="architecture comparison on the sensitive series",
                evidence={"series": ids, "train_size": int(size), "difficulty_scale": scale,
                          "max_spread": round(float(sens[spread_col].max()), 4),
                          "n_model_families": n_fam},
                priority=min(1.0, 0.5 + min(0.5, float(sens[spread_col].max())))))

    # model-AGNOSTIC ceiling: even the BEST family lands in the hard tail and the families agree.
    # "hard for EVERY model" is a claim about models, so it requires at least two of them (ML-10).
    weak = ps[(ps["difficulty_best"] >= hard_q) & (ps[spread_col] <= thr)].head(3)
    if not weak.empty and scale == SCALE_R2_RAW:
        weak = weak[weak["r2_max"] < _POOR_R2]           # keep the legacy absolute floor on R²
    if not weak.empty:
        ids = list(weak.index)
        if n_fam >= 2:
            out.append(GeneratedQuestion(
                id=_qid(dataset, "model_agnostic_ceiling", _fmt(ids)),
                dataset=dataset, kind="model_agnostic_ceiling", difficulty_scale=scale,
                observation=(f"Units [{_fmt(ids)}] are hard for all {n_fam} model families measured "
                             f"at N={size} ({_scale_phrase(scale)}): the best family is still in "
                             "the hard tail and the families agree."),
                hypothesis=("The ceiling is irreducible here — label/replicate noise or a "
                            "biophysical signal absent from all current feature sets."),
                suggested_intervention="estimate the noise ceiling / add a new feature family",
                evidence={"series": ids, "train_size": int(size), "difficulty_scale": scale,
                          "n_model_families": n_fam,
                          "difficulty_best": round(float(weak["difficulty_best"].max()), 4)},
                priority=0.7))
        else:
            fam = sorted(set(df["model"].astype(str))) if "model" in df.columns else ["unknown"]
            out.append(GeneratedQuestion(
                id=_qid(dataset, "single_model_weak", _fmt(ids)),
                dataset=dataset, kind="single_model_weak", difficulty_scale=scale,
                observation=(f"Units [{_fmt(ids)}] are hard for the ONE model family measured "
                             f"({fam[0]}) at N={size} ({_scale_phrase(scale)}). Whether any other "
                             "architecture does better here is UNTESTED — a single family cannot "
                             "support a 'hard for every model' claim (audit ML-10)."),
                hypothesis=(f"Either the ceiling is irreducible, or {fam[0]} simply misses a "
                            "pattern another family would capture; one family cannot tell these "
                            "apart."),
                suggested_intervention="second model family on the weak units",
                evidence={"series": ids, "train_size": int(size), "difficulty_scale": scale,
                          "n_model_families": 1, "model_families": fam,
                          "difficulty_best": round(float(weak["difficulty_best"].max()), 4)},
                priority=0.55))
    return out


def _data_efficiency(df, dataset) -> list[GeneratedQuestion]:
    sizes = sorted(df["train_size"].unique())
    if len(sizes) < 2:
        return []
    lo, hi = sizes[0], sizes[-1]
    pa, pb = _per_unit(df, lo), _per_unit(df, hi)
    scale = _scale_of(pb)
    a, b = pa["difficulty"], pb["difficulty"]
    common = a.index.intersection(b.index)
    if len(common) < _MIN_SERIES:
        return []
    gain = (a.loc[common] - b.loc[common])            # difficulty FELL from smallest to largest N
    if scale != SCALE_R2_RAW:                          # normalise an error-scale gain to a fraction
        denom = a.loc[common].abs().replace(0, np.nan)
        gain = (gain / denom).fillna(0.0)
    improving = gain[gain > _DE_SLOPE].sort_values(ascending=False).head(3)
    if improving.empty:
        return []
    ids = list(improving.index)
    ev = {"series": ids, "from_n": int(lo), "to_n": int(hi), "difficulty_scale": scale,
          "delta_difficulty_max": round(float(improving.max()), 4)}
    if scale == SCALE_R2_RAW:
        ev["delta_r2_max"] = round(float(improving.max()), 4)      # legacy key on the legacy scale
    unit = "ΔR²" if scale == SCALE_R2_RAW else "relative difficulty drop"
    return [GeneratedQuestion(
        id=_qid(dataset, "data_efficiency", _fmt(ids)),
        dataset=dataset, kind="data_efficiency", difficulty_scale=scale,
        observation=(f"Units [{_fmt(ids)}] are still improving from N={lo} to N={hi} "
                     f"({unit} up to {float(improving.max()):.3f}, {_scale_phrase(scale)}): "
                     "data-limited, not model-limited."),
        hypothesis="More labelled data — or transfer from a related dataset — would still help.",
        suggested_intervention="data-efficiency sweep / cross-dataset transfer",
        evidence=ev,
        priority=min(1.0, 0.4 + min(0.6, float(improving.max()))))]


# ------------------------------------------------------------------ the covariate "why" ---
def correlate_difficulty(difficulty: pd.Series, covariates: pd.DataFrame,
                         dataset: str = "unknown", *,
                         scale: str = SCALE_R2_RAW, higher_is_harder: bool = False,
                         alpha: float = _ALPHA, n_perm: int = 10000, seed: int = 0,
                         min_units: int | None = None) -> list[GeneratedQuestion]:
    """Which covariate explains per-unit difficulty? — with a family-wise p-value and no target
    moments.

    `difficulty` is one value per unit (any scale; `higher_is_harder` says which direction is
    hard). `covariates` is a per-unit table. Target-derived columns are DROPPED by name (SA-6) and
    the survivors go through a max-T step-down permutation test over the whole family
    (`statistics.selection`), so the reported correlation carries a family-wise p-value instead of
    being the maximum of three unpenalised attempts (ST-8).

    Emits exactly one `difficulty_covariate` question when the test could RUN — naming the
    covariate if the family-wise test rejects, and reporting the NEGATIVE result if it does not.
    Returns [] when the test could not run at all (too few units, or no testable covariate), since
    there is then nothing to say.
    """
    common = difficulty.index.intersection(covariates.index)
    if len(common) < _MIN_SERIES:
        return []
    y = pd.to_numeric(difficulty.loc[common], errors="coerce").astype(float)
    if not np.isfinite(y).all():
        return []
    rejected_target = {c: "target-derived (audit SA-6): the observed y is not an explanation"
                       for c in covariates.columns if _is_target_derived(c)}
    cols = {c: pd.to_numeric(covariates.loc[common, c], errors="coerce").astype(float).to_numpy()
            for c in covariates.columns if c not in rejected_target}
    res = selection.max_t_correlation(y.to_numpy(), cols, alpha=alpha, n_perm=n_perm, seed=seed,
                                      min_units=min_units)
    if res["refusal"]:
        return []
    dropped = {**rejected_target, **res.get("dropped", {})}
    ev = {"difficulty_scale": scale, "n_series": int(len(common)), "n_units": int(len(common)),
          "candidate_family": res["family"], "n_candidates": res["n_family"],
          "dropped_covariates": dropped,
          "test": "max_t_step_down_permutation", "alpha": alpha,
          "n_permutations": res.get("n_perm"), "exact_permutation": res.get("exact_permutation"),
          "selection_reliability": res.get("reliability"),
          "r_by_covariate": res.get("r_by_covariate", {}),
          "p_max_t_by_covariate": {k: round(float(v), 4) for k, v in res.get("adjusted", {}).items()},
          "reported": bool(res["reported"]), "covariate": res["selected"]}
    if res.get("effect_size_caveat"):
        ev["effect_size_caveat"] = res["effect_size_caveat"]
    scale_note = _scale_phrase(scale)

    if not res["reported"]:
        top = res["ordered"][0] if res.get("ordered") else None
        best_r = res["r_by_covariate"].get(top) if top else None
        p = res.get("p_max_t")
        ev["best_candidate"] = top
        ev["best_candidate_r"] = best_r
        ev["p_max_t"] = round(float(p), 4) if p is not None else None
        return [GeneratedQuestion(
            id=_qid(dataset, "difficulty_covariate", f"none|{','.join(res['family'])}"),
            dataset=dataset, kind="difficulty_covariate", difficulty_scale=scale,
            observation=(f"NO covariate explains per-unit difficulty over {len(common)} units "
                         f"({scale_note}). Best of {res['n_family']} candidates: '{top}' "
                         f"r={best_r:+.2f}, family-wise max-T p={p:.3f} > {alpha}. "
                         f"Dropped as target-derived: {sorted(rejected_target) or 'none'}."),
            hypothesis=("Difficulty here is not explained by the sequence/design covariates "
                        "tested; it needs a new feature family or a noise-ceiling estimate. The "
                        "previously reported expression-moment link was the metric's own "
                        "denominator (audit SA-6), not a cause."),
            suggested_intervention="estimate the noise ceiling / add a new feature family",
            evidence=ev, priority=0.35)]

    col = res["selected"]
    r = float(res["r"])
    p = float(res["p_max_t"])
    harder = (r > 0) == higher_is_harder            # does a HIGHER covariate mean harder?
    direction = "higher" if harder else "lower"
    ev.update({"pearson_r": round(r, 3), "p_max_t": round(p, 4)})
    caveat = ("  Effect size not quotable at this n (" + res["effect_size_caveat"] + ")."
              if res.get("effect_size_caveat") else "")
    return [GeneratedQuestion(
        id=_qid(dataset, "difficulty_covariate", col),
        dataset=dataset, kind="difficulty_covariate", difficulty_scale=scale,
        observation=(f"Per-unit difficulty correlates with '{col}' (r={r:+.2f}, family-wise max-T "
                     f"p={p:.4f} over {res['n_family']} candidates, n={len(common)} units; "
                     f"{scale_note}): units with {direction} '{col}' are the hard ones.{caveat}"),
        hypothesis=f"'{col}' drives predictability; encoding it explicitly should recover the hard units.",
        suggested_intervention="stratify by the correlated covariate",
        evidence=ev,
        priority=min(1.0, 0.5 + abs(r) / 2))]


def dissect_metrics(df: pd.DataFrame, dataset: str = "unknown",
                    train_size: int | None = None) -> list[GeneratedQuestion]:
    """Pure dissection over a per-unit metrics frame (columns: series, model, train_size, r2 and —
    preferred — rmse). Returns generated questions sorted by priority (highest first)."""
    if df.empty or "series" not in df.columns:
        return []
    size = train_size if train_size is not None else int(max(df["train_size"]))
    qs = _difficulty(df, dataset, size) + _sensitivity_and_ceiling(df, dataset, size) \
        + _data_efficiency(df, dataset)
    return sorted(qs, key=lambda q: q.priority, reverse=True)


# ----------------------------------------------------------- sequence/design covariates ---
def _seq_covariates(seqs: pd.Series) -> dict:
    """Covariates for ONE unit, from its sequences alone — nothing here touches the target."""
    s = seqs.astype(str).str.upper()
    gc = s.apply(lambda x: (x.count("G") + x.count("C")) / max(1, len(x)))
    lens = s.str.len()
    km = _kmer_matrix(s.tolist(), 3)
    centroid = km.mean(axis=0, keepdims=True)
    return {"gc": float(gc.mean()),
            "gc_sd": float(gc.std(ddof=0)) if len(gc) > 1 else 0.0,
            "length_cv": float(lens.std(ddof=0) / lens.mean()) if lens.mean() else 0.0,
            "seq_entropy": _position_entropy(s.tolist()),
            "kmer_dispersion": float(np.linalg.norm(km - centroid, axis=1).mean()),
            "n_rows": int(len(s))}


def _position_entropy(sequences) -> float:
    """Mean per-position Shannon entropy (bits) over the unit's sequences, truncated to the
    shortest. 0 = every sequence identical at every position; 2 = uniform over ACGT."""
    if not sequences:
        return 0.0
    m = min(len(s) for s in sequences)
    if m == 0 or len(sequences) < 2:
        return 0.0
    arr = np.frombuffer("".join(s[:m] for s in sequences).encode("ascii", "replace"),
                        dtype="S1").reshape(len(sequences), m)
    tot = 0.0
    for base in (b"A", b"C", b"G", b"T"):
        p = (arr == base).mean(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            tot = tot - np.where(p > 0, p * np.log2(p), 0.0)
    return float(np.mean(tot))


def _train_distance(unit_seqs, train_km_centroid) -> float:
    """Mean distance from the unit's 3-mer composition to the TRAINING set's centroid — a
    sequence-derived proxy for "is this unit outside what the model was trained on?". Uses train X
    only; no target values enter."""
    km = _kmer_matrix([str(s) for s in unit_seqs], 3)
    return float(np.linalg.norm(km - train_km_centroid, axis=1).mean())


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
    """Per-series covariates keyed by the SAME series id as the baseline metrics — so
    `correlate_difficulty` can answer 'why are the hard series hard?'.

    SEQUENCE- AND DESIGN-DERIVED ONLY. The pre-2026-11-02 version also returned `expr_mean` and
    `expr_std`; `expr_std` is R²'s own denominator and was the covariate the layer reported for
    E. coli (audit SA-6), so the target column is no longer read here at all.

    Best-effort: any failure returns None so the pure dissections still run (the harness never
    depends on it)."""
    try:
        from ..data import datasets
        if not datasets.data_present(dataset):
            return None
        from ..data.cleaning import SEQ_COL, SERIES_COL
        frame = _covariate_frame(dataset)
        if frame is None or SERIES_COL not in frame.columns:
            return None
        rows = {}
        for sid, g in frame.groupby(SERIES_COL):
            key = int(sid) if str(sid).lstrip("-").isdigit() else sid
            rows[key] = _seq_covariates(g[SEQ_COL])
        return pd.DataFrame.from_dict(rows, orient="index") if rows else None
    except Exception:
        return None


def dissect_dataset(dataset: str, metrics_csv, train_size: int | None = None) -> list[GeneratedQuestion]:
    """Load a registry metrics.csv for `dataset`, run the pure dissections, and (when the raw data is
    present) add the covariate-correlation question. `metrics_csv` is a path to the baseline run's
    metrics.csv (e.g. experiments/runs/<baseline_run_id>/metrics.csv). A missing path means the
    dataset has no COMPLETE baseline registry yet, which yields no questions rather than an error —
    and never a silent fall back to an older registry's numbers (ML-02)."""
    if metrics_csv is None or not Path(metrics_csv).exists():
        return []
    df = pd.read_csv(metrics_csv)
    qs = dissect_metrics(df, dataset, train_size)
    cov = _series_covariates(dataset)
    if cov is not None:
        size = train_size if train_size is not None else int(max(df["train_size"]))
        ps = _per_unit(df, size)
        qs += correlate_difficulty(ps["difficulty"], cov, dataset, scale=_scale_of(ps),
                                   higher_is_harder=True)
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
    p = _registry_metrics(dataset)
    return p if p and p.exists() else None


def _insight_path(dataset: str) -> Path:
    return ROOT / "experiments" / "insights" / f"{dataset}.jsonl"


def save_questions(dataset: str, questions: list[GeneratedQuestion]) -> Path:
    p = _insight_path(dataset)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(q.model_dump_json() for q in questions), encoding="utf-8")
    return p


def load_questions(dataset: str) -> list[GeneratedQuestion]:
    """Persisted generated questions for a dataset (written by scripts/run_insight.py), or [].

    Questions written before the 2026-11-02 rebuild have no `difficulty_scale` field; they
    deserialize as `r2_raw_legacy` and are flagged `evidence["pre_rebuild"] = True` so a consumer
    can see that their difficulty is raw R² and their covariate (if any) carries no family-wise
    p-value. A line that cannot be parsed at all is SKIPPED rather than raising, so an old or
    half-written insight file degrades to fewer questions instead of breaking the loop."""
    p = _insight_path(dataset)
    if not p.exists():
        return []
    out: list[GeneratedQuestion] = []
    for ln in p.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        try:
            q = GeneratedQuestion.model_validate_json(ln)
        except Exception:
            continue
        if q.pre_rebuild:
            q.evidence = {**q.evidence, "pre_rebuild": True,
                          "pre_rebuild_note": ("written before the SA-6/ST-8/ML-10 rebuild: "
                                               "difficulty is raw R², and any covariate is an "
                                               "unpenalised max |r| with no family-wise p-value")}
        out.append(q)
    return out


def pooled_predictions(dataset: str, *, model_family: str = "rf", feature_set: str = "one_hot",
                       seed: int = 1, frac: float = 0.1, max_train: int = 6000):
    """A pooled baseline's held-out predictions via the HARNESS pooled path (not a shadow pipeline):
    returns (test_sequences, y_true, y_pred), positionally aligned. Deterministic given the seed;
    train rows capped for speed. Kept for readers that only need one family's predictions —
    `pooled_probe` is what the dissection uses, because it also returns the TRAIN sequences the
    clusterer has to be fitted on and more than one model family."""
    probe = pooled_probe(dataset, families=(model_family,), feature_set=feature_set, seed=seed,
                         frac=frac, max_train=max_train)
    return probe["test_sequences"], probe["y_true"], probe["predictions"][model_family]


def pooled_probe(dataset: str, *, families=("rf", "mlp"), feature_set: str = "one_hot",
                 seed: int = 1, frac: float = 0.1, max_train: int = 6000) -> dict:
    """Pooled held-out predictions from MORE THAN ONE model family, plus the TRAIN sequences.

    Two things the pre-rebuild path could not do (audit ML-10): a "hard for every model" claim needs
    at least two families, and the neighbourhood clusterer has to be fitted on train sequences
    rather than on the test sequences it then judges. Returns
    {train_sequences, test_sequences, y_true, predictions: {family: y_pred}, families}.
    """
    from types import SimpleNamespace

    from ..data.cleaning import SEQ_COL, TARGET_COL
    from ..experiments import pooled_runner as P
    spec = SimpleNamespace(dataset=dataset, seed=seed, feature_set=feature_set,
                           feature_scaling="none", hyperparameters={}, sampling_policy="random")
    train_full, test = P.holdout(spec, frac=frac)
    train = P.subsample(train_full, min(max_train, len(train_full)), "random", seed)
    preds = {}
    for fam in families:
        try:
            preds[fam] = P.fit_predict(spec, fam, train, test)
        except Exception:
            continue
    return {"train_sequences": train[SEQ_COL].tolist(), "test_sequences": test[SEQ_COL].tolist(),
            "y_true": test[TARGET_COL].to_numpy(), "predictions": preds,
            "families": list(preds)}


def dissect_any(dataset: str, *, k: int = 8, train_size: int | None = None,
                families=("rf", "mlp")) -> list[GeneratedQuestion]:
    """Structure-aware dissection. Per-series datasets (E. coli) dissect their baseline registry
    metrics; pooled datasets (yeast, dream2022, ...) DISCOVER neighborhoods from a real pooled
    baseline's held-out predictions and dissect those. [] when neither is available."""
    path = default_metrics_path(dataset)
    if path:
        return dissect_dataset(dataset, path, train_size=train_size)
    try:
        from ..data import datasets as ds
        if ds.exists(dataset) and ds.spec(dataset).structure == "pooled" and ds.data_present(dataset):
            probe = pooled_probe(dataset, families=families)
            if not probe["predictions"]:
                return []
            return dissect_pooled_predictions(
                probe["test_sequences"], probe["y_true"], probe["predictions"], dataset, k=k,
                train_sequences=probe["train_sequences"])
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
            datasets = list(_DEFAULT_DATASETS)
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
# cluster as a series-like unit, then dissect exactly as for E. coli.
#
# The clusterer is FITTED ON TRAIN SEQUENCES and applied to the test sequences (audit ML-10): a
# partition chosen on the same rows it is then scored on is a partition chosen partly for its
# errors. `cluster_sequences` keeps the old in-sample behaviour for callers that have no train set,
# and the questions it produces are stamped `cluster_fit="test_sequences"` so that is visible.
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


class SequenceClusterer:
    """A neighbourhood partition fitted on one set of sequences and APPLIED to another."""

    def __init__(self, model, kmer: int, k: int):
        self.model, self.kmer, self.k = model, kmer, k

    def predict(self, sequences) -> np.ndarray:
        n = len(sequences)
        if n == 0:
            return np.array([], dtype=int)
        if self.model is None:
            return np.zeros(n, dtype=int)
        return self.model.predict(_kmer_matrix(sequences, self.kmer)).astype(int)


def fit_sequence_clusters(train_sequences, k: int = 8, *, seed: int = 0,
                          kmer: int = 3) -> SequenceClusterer:
    """Fit k neighbourhoods on TRAIN sequences (length-normalized k-mer composition -> KMeans).
    Deterministic for a given seed. Clamps k to the sample size; k<=1 -> a single neighbourhood."""
    n = len(train_sequences)
    k = max(1, min(k, n))
    if n == 0 or k == 1:
        return SequenceClusterer(None, kmer, 1)
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(_kmer_matrix(train_sequences, kmer))
    return SequenceClusterer(km, kmer, k)


def cluster_sequences(sequences, k: int = 8, *, seed: int = 0, kmer: int = 3) -> np.ndarray:
    """IN-SAMPLE partition: fit k neighborhoods on `sequences` and label those same sequences.

    Prefer `fit_sequence_clusters(train_sequences).predict(test_sequences)`. This is the fallback
    for callers with no separate train set, and it is the behaviour audit ML-10 flagged ("clusters
    are fitted on test sequences and judged by test R²")."""
    n = len(sequences)
    if n == 0:
        return np.array([], dtype=int)
    return fit_sequence_clusters(sequences, k=k, seed=seed, kmer=kmer).predict(sequences)


def _r2_rmse(y_true, y_pred):
    from ..statistics.bootstrap import _r2
    err = np.asarray(y_true, float) - np.asarray(y_pred, float)
    return _r2(y_true, y_pred), float(np.sqrt(np.mean(err ** 2)))


def neighborhoods_to_metrics_frame(labels, y_true, y_pred, *, model: str = "pooled",
                                   train_size: int = 0, min_n: int = 20,
                                   noise_rmse=None) -> pd.DataFrame:
    """Per-neighborhood metrics from a pooled model's held-out predictions -> a dissect-compatible
    frame (series = neighborhood id). Emits BOTH `r2` and `rmse`; the dissection prefers `rmse`
    because per-cluster R² divides by the cluster's own Var(y) (audit ML-10/SA-6).

    `y_pred` may be a single array (one model family) or a {family: array} mapping — "hard for every
    model" needs at least two. `noise_rmse` is an optional per-unit noise-ceiling RMSE (see
    `replicate_noise_rmse`) which puts difficulty on `SCALE_NOISE`. Neighborhoods with < min_n test
    points are dropped (unstable metrics)."""
    y_true, labels = np.asarray(y_true, float), np.asarray(labels)
    preds = y_pred if isinstance(y_pred, dict) else {model: y_pred}
    rows = []
    for c in np.unique(labels):
        m = labels == c
        if int(m.sum()) < min_n:
            continue
        for fam, pred in preds.items():
            r2, rmse = _r2_rmse(y_true[m], np.asarray(pred, float)[m])
            row = {"iteration": "iteration_1", "series": int(c), "model": str(fam),
                   "train_size": int(train_size), "r2": r2, "rmse": rmse, "n": int(m.sum())}
            if noise_rmse:
                nz = noise_rmse.get(int(c))
                if nz:
                    row["rmse_noise"] = float(nz)
            rows.append(row)
    frame = pd.DataFrame(rows)
    # a partial noise map would silently mix two scales across units -> keep it all-or-nothing
    if "rmse_noise" in frame.columns and frame["rmse_noise"].isna().any():
        frame = frame.drop(columns=["rmse_noise"])
    return frame


def _neighborhood_covariates(sequences, labels, y_true=None, *, train_sequences=None) -> pd.DataFrame:
    """Per-neighborhood covariates keyed by cluster id — the pooled analogue of `_series_covariates`.

    SEQUENCE- AND DESIGN-DERIVED ONLY. `y_true` is accepted and IGNORED (the pre-rebuild version
    returned `expr_mean`/`expr_std` from it, which is audit SA-6). When `train_sequences` is given,
    adds `train_kmer_dist` — how far the cluster sits from the training distribution in k-mer space,
    computed from train X only."""
    labels = np.asarray(labels)
    seqs = pd.Series([str(s).upper() for s in sequences])
    centroid = None
    if train_sequences is not None and len(train_sequences):
        centroid = _kmer_matrix([str(s) for s in train_sequences], 3).mean(axis=0, keepdims=True)
    rows = {}
    for c in np.unique(labels):
        m = labels == c
        cov = _seq_covariates(seqs[m])
        if centroid is not None:
            cov["train_kmer_dist"] = _train_distance(seqs[m].tolist(), centroid)
        rows[int(c)] = cov
    return pd.DataFrame.from_dict(rows, orient="index")


# ------------------------------------------------------------- variance-matched cluster null ---
def y_matched_partition_null(labels, y_true, y_pred, *, n_draw: int = 2000, n_bins: int = 10,
                             seed: int = 0, units=None) -> dict:
    """Is this cluster harder than a RANDOM partition of the same rows with the same y profile?

    The question audit ML-10 asks. Each draw is one genuine random partition of the same test rows
    into parts that have, for every unit, exactly the same number of rows in every y-quantile bin as
    the real unit — so each null part matches its unit in size and (up to bin resolution) in mean
    and variance of y. Per-unit RMSE is recomputed on each null partition, and the units are ranked
    against that null with a family-wise step-down max-T test, because picking the worst of k
    clusters is itself a selection (ST-8).

    Returns {units: {unit: {rmse, p_max_t, null_rmse_median, var_ratio, n}}, n_draw, n_bins,
    selected, harder_than_null: [units with p <= alpha]}.
    """
    labels = np.asarray(labels)
    y = np.asarray(y_true, float)
    pred = np.asarray(y_pred, float)
    keep = np.isin(labels, list(units)) if units is not None else np.ones(len(labels), bool)
    labels, y, pred = labels[keep], y[keep], pred[keep]
    uniq = [int(u) for u in np.unique(labels)]
    out: dict = {"units": {}, "n_draw": int(n_draw), "n_bins": int(n_bins),
                 "selected": None, "harder_than_null": []}
    if len(uniq) < 2 or len(y) < 4:
        return out
    e2 = (y - pred) ** 2
    nb = max(2, min(int(n_bins), len(y) // max(2, len(uniq))))
    try:
        bins = pd.qcut(y, q=nb, labels=False, duplicates="drop")
    except Exception:
        bins = np.zeros(len(y), dtype=int)
    bins = np.asarray(bins)
    rng = np.random.default_rng(seed)

    sums = {u: np.zeros(n_draw) for u in uniq}
    ysum = {u: np.zeros(n_draw) for u in uniq}
    ysq = {u: np.zeros(n_draw) for u in uniq}
    counts = {u: int((labels == u).sum()) for u in uniq}
    for b in np.unique(bins):
        idx = np.flatnonzero(bins == b)
        if idx.size == 0:
            continue
        # how many rows of this bin each unit really owns -> deal exactly that many at random
        own = [int(((labels == u) & (bins == b)).sum()) for u in uniq]
        order = np.argsort(rng.random((n_draw, idx.size)), axis=1)
        e2b = np.cumsum(e2[idx][order], axis=1)
        yb = np.cumsum(y[idx][order], axis=1)
        y2b = np.cumsum((y[idx] ** 2)[order], axis=1)
        pos = 0
        for u, cnt in zip(uniq, own):
            if cnt:
                a, z = pos, pos + cnt
                sums[u] += e2b[:, z - 1] - (e2b[:, a - 1] if a else 0.0)
                ysum[u] += yb[:, z - 1] - (yb[:, a - 1] if a else 0.0)
                ysq[u] += y2b[:, z - 1] - (y2b[:, a - 1] if a else 0.0)
            pos += cnt

    observed, nulls = {}, {}
    for u in uniq:
        n_u = counts[u]
        if n_u < 2:
            continue
        observed[u] = float(np.sqrt(np.mean(e2[labels == u])))
        nulls[u] = np.sqrt(sums[u] / n_u)
    if len(observed) < 2:
        return out
    res = selection.step_down_max_t({str(u): v for u, v in observed.items()},
                                    {str(u): nulls[u] for u in observed}, alpha=_ALPHA)
    for u in observed:
        n_u = counts[u]
        var_obs = float(np.var(y[labels == u]))
        var_null = ysq[u] / n_u - (ysum[u] / n_u) ** 2
        ratio = (float(np.median(var_null)) / var_obs) if var_obs else None
        # the y-marginal is matched only at BIN resolution: a small unit whose y sits inside one
        # bin can still end up with a null part of noticeably different variance. Flag it rather
        # than pretend the match is exact. ratio > 1 makes the null HARDER, i.e. conservative.
        match = "ok" if ratio is None or 0.8 <= ratio <= 1.25 else (
            "loose_conservative" if ratio > 1.25 else "loose_liberal")
        out["units"][u] = {
            "n": n_u, "rmse": round(observed[u], 4),
            "null_rmse_median": round(float(np.median(nulls[u])), 4),
            "p_max_t": round(float(res["adjusted"][str(u)]), 4),
            "var_y": round(var_obs, 4),
            "var_ratio_null_over_observed": round(ratio, 3) if ratio is not None else None,
            "variance_match": match,
        }
    out["harder_than_null"] = [u for u in observed if out["units"][u]["p_max_t"] <= _ALPHA
                               and observed[u] > float(np.median(nulls[u]))]
    out["harder_than_null"].sort(key=lambda u: out["units"][u]["p_max_t"])
    out["selected"] = out["harder_than_null"][0] if out["harder_than_null"] else None
    out["variance_match_loose"] = [u for u in out["units"]
                                   if out["units"][u]["variance_match"] != "ok"]
    return out


def dissect_pooled_predictions(sequences, y_true, y_pred, dataset: str = "unknown", *,
                               k: int = 8, seed: int = 0, min_n: int = 20,
                               train_sequences=None, noise_rmse=None,
                               n_draw: int = 2000) -> list[GeneratedQuestion]:
    """Discover neighborhoods in a pooled dataset and dissect them.

    `y_pred` may be one array or a {family: array} mapping; two or more families are what lets a
    "hard for every model" question be emitted at all (audit ML-10). When `train_sequences` is
    given the clusterer is fitted on TRAIN sequences and applied to the test sequences; otherwise it
    falls back to the in-sample partition and every question is stamped
    `cluster_fit="test_sequences"`.

    Difficulty is per-cluster RMSE, and the clusters are additionally ranked against a y-matched
    random partition (`y_matched_partition_null`) so "hardest" means "larger error than a random
    part of the same size and the same y profile" rather than "smallest Var(y)".
    """
    if train_sequences is not None and len(train_sequences):
        clusterer = fit_sequence_clusters(train_sequences, k=k, seed=seed)
        labels = clusterer.predict(sequences)
        cluster_fit = "train_sequences"
    else:
        labels = cluster_sequences(sequences, k=k, seed=seed)
        cluster_fit = "test_sequences"
    preds = y_pred if isinstance(y_pred, dict) else {"pooled": y_pred}
    frame = neighborhoods_to_metrics_frame(labels, y_true, preds, train_size=len(y_true),
                                           min_n=min_n, noise_rmse=noise_rmse)
    qs = dissect_metrics(frame, dataset)
    null = {"units": {}, "harder_than_null": []}
    if not frame.empty:
        kept = sorted(set(int(c) for c in frame["series"]))
        # rank against a variance-matched null, using the mean prediction across families
        mean_pred = np.mean([np.asarray(p, float) for p in preds.values()], axis=0)
        null = y_matched_partition_null(labels, y_true, mean_pred, units=kept, seed=seed,
                                        n_draw=n_draw)
        ranked = _null_ranked_difficulty(dataset, null, frame, cluster_fit, len(preds))
        if ranked:
            # the null-ranked verdict SUPERSEDES the quantile-threshold one: "top quartile of RMSE"
            # names a hardest unit even when no unit beats a matched random partition (audit ML-10)
            qs = [q for q in qs if q.kind != "series_difficulty"] + ranked
        cov = _neighborhood_covariates(sequences, labels, train_sequences=train_sequences)
        cov = cov.loc[[c for c in cov.index if c in kept]]
        ps = _per_unit(frame, int(len(y_true)))
        qs += correlate_difficulty(ps["difficulty"], cov, dataset, scale=_scale_of(ps),
                                   higher_is_harder=True, seed=seed)
    for q in qs:                            # mark that the unit was discovered, not a natural series
        q.evidence["unit"] = "discovered_neighborhood"
        q.evidence["cluster_fit"] = cluster_fit
        q.evidence["n_model_families"] = q.evidence.get("n_model_families", len(preds))
        if cluster_fit == "test_sequences":
            q.evidence["cluster_fit_caveat"] = (
                "the partition was fitted on the SAME test sequences it is scored on (no train "
                "sequences supplied); pass train_sequences to remove this (audit ML-10)")
        if null["units"]:
            q.evidence["null_ranking"] = {"test": "y_matched_random_partition",
                                          "harder_than_null": null["harder_than_null"],
                                          "n_draw": null["n_draw"]}
    return sorted(qs, key=lambda q: q.priority, reverse=True)


def _null_ranked_difficulty(dataset, null, frame, cluster_fit, n_fam) -> list[GeneratedQuestion]:
    """The ML-10 replacement for "hardest cluster by R²": the clusters whose RMSE beats a y-matched
    random partition of the same size, or an explicit statement that none does."""
    if not null["units"]:
        return []
    hard = null["harder_than_null"]
    ev = {"difficulty_scale": SCALE_RMSE, "test": "y_matched_random_partition",
          "alpha": _ALPHA, "n_draw": null["n_draw"], "n_bins": null["n_bins"],
          "per_unit": null["units"], "n_model_families": int(n_fam),
          "cluster_fit": cluster_fit}
    if not hard:
        worst = max(null["units"], key=lambda u: null["units"][u]["rmse"])
        ev["series"] = []
        ev["highest_rmse_unit"] = worst
        return [GeneratedQuestion(
            id=_qid(dataset, "series_difficulty", f"null|{sorted(null['units'])}"),
            dataset=dataset, kind="series_difficulty", difficulty_scale=SCALE_RMSE,
            observation=(f"No neighbourhood is harder than a random partition of the same size and "
                         f"the same y profile ({len(null['units'])} units, "
                         f"{null['n_draw']} draws, family-wise max-T). The highest-RMSE unit "
                         f"({worst}) has p={null['units'][worst]['p_max_t']:.3f}. Ranking these "
                         "units by per-cluster R² would still have named a 'hardest' one, because "
                         "R² divides by each cluster's own Var(y) (audit ML-10)."),
            hypothesis=("The apparent per-cluster difficulty spread is accounted for by cluster "
                        "size and target distribution; there is no neighbourhood effect to chase "
                        "at this k."),
            suggested_intervention="estimate the noise ceiling / add a new feature family",
            evidence=ev, priority=0.3)]
    ids = hard[:3]
    ev["series"] = [int(u) for u in ids]
    ev["variance_match_loose"] = null.get("variance_match_loose", [])
    best_p = null["units"][ids[0]]["p_max_t"]
    detail = "; ".join(f"{u}: RMSE {null['units'][u]['rmse']:.3f} vs null median "
                       f"{null['units'][u]['null_rmse_median']:.3f}, p={null['units'][u]['p_max_t']:.3f}"
                       for u in ids)
    loose = [u for u in ids if null["units"][u]["variance_match"] != "ok"]
    if loose:
        detail += ("; the y-marginal match is only approximate for unit(s) "
                   + ", ".join(f"{u} (null/observed Var(y)="
                               f"{null['units'][u]['var_ratio_null_over_observed']}, "
                               f"{null['units'][u]['variance_match']})" for u in loose))
    return [GeneratedQuestion(
        id=_qid(dataset, "series_difficulty", f"null|{_fmt(ids)}"),
        dataset=dataset, kind="series_difficulty", difficulty_scale=SCALE_RMSE,
        observation=(f"Units [{_fmt(ids)}] have larger RMSE than a random partition of the same "
                     f"size and the same y profile ({detail}; {null['n_draw']} draws, family-wise "
                     f"max-T over {len(null['units'])} units)."),
        hypothesis=("These neighbourhoods carry a sequence property the current features miss — "
                    "not merely a smaller target variance."),
        suggested_intervention="dissect covariates / add a mechanistic feature",
        evidence=ev, priority=min(1.0, 0.6 + (0.3 if best_p <= 0.01 else 0.0)))]
