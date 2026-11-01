"""Generic pooled runner (K6) — the spec-driven generalization of the yeast runner.

Handles any `structure: pooled` dataset: sources the cleaned frame via its adapter, splits
(provided held-out or per-group / target-stratified holdout), subsamples to a train size, and
returns per-sequence predictions for a SEQUENCE-LEVEL paired bootstrap. Length comes from the
registry; the dataset is passed explicitly to the feature pipeline (K6 — no length inference).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from ..data import adapters, datasets, families
from ..data.cleaning import SEQ_COL, SERIES_COL, TARGET_COL

FAMILY_COL = "design_family"       # travels on the frame so filtered subregions keep the grouping
from ..models import registry as model_registry
from ..training.reproducibility import set_seed
from ..training.train import features_for


@lru_cache(maxsize=4)
def _frame(dataset_id: str) -> pd.DataFrame:
    from ..data import datasets
    return datasets.apply_target_transform(adapters.frame_for(dataset_id), dataset_id)


def grouped_holdout(df: pd.DataFrame, families: pd.Series, *, frac: float = 0.1, seed: int = 0):
    """(train, test) with whole FAMILIES on one side — the fix for audit SA-1/ML-03.

    A per-group stratified holdout takes a slice out of every family, which guarantees that each
    held-out sequence has a sibling in train; on yeast that inflates CNN R² from 0.114 to 0.900.
    Families are shuffled and assigned to test until the row target is met, so the test fraction is
    approximate by construction (a family is indivisible)."""
    rng = np.random.default_rng(seed)
    groups = {}
    for key, idx in df.groupby(families.reindex(df.index), sort=True).groups.items():
        groups[key] = np.asarray(idx)
    order = rng.permutation(list(groups))
    target = max(1, int(round(frac * len(df))))
    test_idx: list = []
    for key in order:
        if len(test_idx) >= target:
            break
        test_idx += list(groups[key])
    if not test_idx or len(test_idx) >= len(df):        # degenerate (one family, or all of them)
        return holdout_frame(df.drop(columns=[FAMILY_COL], errors="ignore"), frac=frac, seed=seed)
    test = df.loc[test_idx]
    train = df.drop(index=test_idx)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def holdout_frame(df: pd.DataFrame, *, frac: float = 0.1, seed: int = 0):
    """(train, test) split of an arbitrary frame. Provided 'split' column respected; else, if the
    frame carries a design-family column, whole families are held out; else a stratified holdout
    (per-group if a series column exists, else target-quantile). Used by the generic holdout AND by
    C6 subregion tournaments that filter the frame first — which is why the family column travels
    on the frame: a filtered subregion must still respect the grouping."""
    if "split" in df.columns:
        tr = df[df["split"] == "train"].drop(columns=["split"]).reset_index(drop=True)
        te = df[df["split"] == "test"].drop(columns=["split"]).reset_index(drop=True)
        if len(tr) and len(te):
            return tr, te
        df = df.drop(columns=["split"])                # provided split empty after filtering -> re-split
    if FAMILY_COL in df.columns:
        return grouped_holdout(df, df[FAMILY_COL], frac=frac, seed=seed)
    rng = np.random.default_rng(seed)
    if SERIES_COL in df.columns:
        group = df[SERIES_COL]
    else:
        group = pd.qcut(df[TARGET_COL], q=min(10, max(2, len(df) // 50)), labels=False,
                        duplicates="drop")
    test_idx: list = []
    for _, grp in df.groupby(group):
        k = max(1, int(round(frac * len(grp))))
        test_idx += list(rng.choice(grp.index.to_numpy(), size=min(k, len(grp)), replace=False))
    test = df.loc[test_idx]
    train = df.drop(index=test_idx)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def with_families(df: pd.DataFrame, ds) -> tuple[pd.DataFrame, str]:
    """Attach the design-family label (and say how it was derived) when the dataset declares one."""
    labels, how = families.family_labels(df, ds, seq_col=SEQ_COL)
    if labels is None:
        return df, how
    return df.assign(**{FAMILY_COL: labels.to_numpy()}), how


def scope_frame(dataset_id: str) -> pd.DataFrame:
    """The dataset's frame with design families already attached (when it declares them).

    This is the frame a C6 subregion filter must start from: `strata.filter` keeps whole rows, so
    the family label survives the filter and `holdout_frame` can still hold whole families out.
    Filtering the bare frame instead would silently put a subregion tournament back on the
    stratified split the grouped-holdout fix removed (audit SA-1/ML-03)."""
    ds = datasets.spec(dataset_id)
    df = _frame(dataset_id)
    if ds.split_strategy == "grouped_holdout":
        df, _ = with_families(df, ds)
    return df


def families_of(frame: pd.DataFrame):
    """The design-family label per row, or None when the frame carries no grouping — i.e. the
    `groups=` argument for the paired bootstrap, whose unit must be the family (audit ST-7)."""
    return frame[FAMILY_COL].to_numpy() if FAMILY_COL in frame.columns else None


def holdout(spec, frac: float = 0.1):
    """(train, test) frames. Provided split if the adapter supplied one; else `grouped_holdout`
    when the dataset declares design families; else a stratified holdout (per-group if a series
    column exists, else target-quantile)."""
    df = _frame(spec.dataset)
    ds = datasets.spec(spec.dataset)
    if ds.split_strategy == "provided" and "split" in df.columns:
        tr = df[df["split"] == "train"].drop(columns=["split"]).reset_index(drop=True)
        te = df[df["split"] == "test"].drop(columns=["split"]).reset_index(drop=True)
        return tr, te
    if ds.split_strategy == "grouped_holdout":
        df, _ = with_families(df, ds)
    return holdout_frame(df, frac=frac, seed=spec.seed)


def split_report(spec, frac: float = 0.1) -> dict:
    """What the split actually did — recorded on the run so a grouping is never implicit, and so a
    result computed under `grouped_holdout` can never be silently compared with an older one."""
    ds = datasets.spec(spec.dataset)
    df = _frame(spec.dataset)
    df, how = with_families(df, ds)
    train, test = holdout(spec, frac=frac)
    out = {"split_strategy": ds.split_strategy, "family_source": how,
           "n_train": int(len(train)), "n_test": int(len(test)),
           "test_frac": round(len(test) / max(1, len(train) + len(test)), 4)}
    if FAMILY_COL in df.columns:
        out["n_families"] = int(df[FAMILY_COL].nunique())
        out["families_test"] = int(test[FAMILY_COL].nunique()) if FAMILY_COL in test.columns else None
        out["families_shared"] = int(len(set(train.get(FAMILY_COL, pd.Series(dtype=object)))
                                         & set(test.get(FAMILY_COL, pd.Series(dtype=object)))))
    return out


def subsample(train: pd.DataFrame, size: int, policy: str, seed: int) -> pd.DataFrame:
    if size >= len(train):
        return train
    rng = np.random.default_rng(seed)
    if policy == "expression_stratified":
        q = pd.qcut(train[TARGET_COL], q=min(10, max(2, size // 50)), labels=False, duplicates="drop")
        idx: list = []
        for _, grp in train.groupby(q):
            k = max(1, int(round(size * len(grp) / len(train))))
            idx += list(rng.choice(grp.index.to_numpy(), size=min(k, len(grp)), replace=False))
        return train.loc[idx[:size]].reset_index(drop=True)
    idx = rng.choice(train.index.to_numpy(), size=size, replace=False)
    return train.loc[idx].reset_index(drop=True)


def fit_predict(spec, model_family: str, train: pd.DataFrame, test: pd.DataFrame):
    length = datasets.seq_len(spec.dataset)
    set_seed(spec.seed)
    Xtr = features_for(model_family, train, spec.feature_set, length, spec.dataset)
    Xte = features_for(model_family, test, spec.feature_set, length, spec.dataset)
    if spec.feature_scaling not in (None, "none") and model_registry.feature_kind(model_family) == "flat":
        from ..features import scaling as sc
        name = sc.recommend_scaler(Xtr)[0] if spec.feature_scaling == "auto" else spec.feature_scaling
        scaler = sc.make_scaler(name)
        if scaler is not None:
            scaler.fit(Xtr)
            Xtr, Xte = scaler.transform(Xtr), scaler.transform(Xte)
    model = model_registry.make(model_family, seed=spec.seed,
                                hyperparameters=spec.hyperparameters or None)
    model.fit(Xtr, train[TARGET_COL].to_numpy())
    return model.predict(Xte)


def run_pooled(spec, *, model_family: str | None = None, frac: float = 0.1) -> dict:
    """Train `model_family` pooled at each train_size; return per-sequence predictions on a fixed
    held-out test set. {y_test, test_series, preds:{size:ndarray}, n_train_full}."""
    mf = model_family or spec.model_family
    train_full, test = holdout(spec, frac=frac)
    preds = {}
    for size in sorted(set(spec.train_sizes)):
        sub = subsample(train_full, size, spec.sampling_policy, spec.seed)
        preds[size] = fit_predict(spec, mf, sub, test)
    return {"y_test": test[TARGET_COL].to_numpy(),
            "test_series": test[SERIES_COL].to_numpy() if SERIES_COL in test.columns else None,
            # Design families of the held-out rows: the correct resampling unit for a dataset whose
            # members are not independent (a 4-member family is not 4 observations).
            "test_families": families_of(test),
            "preds": preds, "n_train_full": int(len(train_full))}
