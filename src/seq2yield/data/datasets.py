"""Dataset registry (K6) — the per-dataset frozen contract that makes onboarding declarative.

A `DatasetSpec` captures everything the harness/council need to treat a new sequence→function
dataset uniformly: sequence length, structure (per-series vs pooled), bootstrap unit, readout,
split strategy, and which models/features/embedders apply. Specs live in `configs/datasets/*.yaml`
so adding dataset #N is a config + adapter change — never an edit to a strict protected file.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = ROOT / "configs" / "datasets"


class DatasetSpec(BaseModel):
    id: str
    display_name: str = ""
    organism: str = "unknown"
    modality: str = "coding"                 # coding | promoter | utr | rbs | splice | polya | ires
    seq_len: int
    alphabet: str = "ACGT"
    target_col: str = "Protein"              # raw source column the adapter reads
    target_transform: str = "none"           # none | standardize | logit | log1p
    structure: str = "per_series"            # per_series | pooled
    bootstrap_unit: str = "series"           # series | sequence  (C3 fence)
    throughput_floor: int = 10000
    adapter: str | None = None               # module in data/adapters/ (None = built-in path)
    split_strategy: str = "provided"         # provided | stratified_holdout
    applicable_models: list[str] = Field(default_factory=lambda: ["rf", "mlp", "cnn"])
    applicable_feature_sets: list[str] = Field(default_factory=lambda: ["one_hot", "kmer"])
    applicable_embedders: list[str] = Field(default_factory=list)
    strata: list[str] = Field(default_factory=list)   # C6: subregion axes ([] -> modality default)
    citation: str = ""
    license: str = ""
    source: dict = Field(default_factory=dict)


@lru_cache(maxsize=1)
def _load_all() -> dict[str, DatasetSpec]:
    specs: dict[str, DatasetSpec] = {}
    if CONFIG_DIR.exists():
        for p in sorted(CONFIG_DIR.glob("*.yaml")):
            d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            s = DatasetSpec(**d)
            specs[s.id] = s
    return specs


def all_ids() -> list[str]:
    return list(_load_all().keys())


def exists(dataset_id: str) -> bool:
    return dataset_id in _load_all()


def spec(dataset_id: str) -> DatasetSpec:
    specs = _load_all()
    if dataset_id not in specs:
        raise KeyError(f"unknown dataset '{dataset_id}'. registered: {list(specs)}")
    return specs[dataset_id]


def seq_len(dataset_id: str) -> int:
    """Sequence length for a dataset (the length feature builders must use). Defaults to 96 for an
    unregistered id so legacy callers never crash mid-refactor."""
    s = _load_all().get(dataset_id)
    return s.seq_len if s else 96


def applicable_feature_sets(dataset_id: str) -> list[str]:
    s = _load_all().get(dataset_id)
    return s.applicable_feature_sets if s else ["one_hot", "kmer", "mechanistic", "mixed"]


def apply_target_transform(df, dataset_id: str):
    """Apply the dataset's readout transform to TARGET_COL (K6). Parameter-free + elementwise so it
    never leaks train/test: log1p for skewed positives, logit for bounded (0,1) fractions
    (splicing PSI / APA isoform), none/standardize are R²-invariant (affine) -> left as-is."""
    import numpy as np

    from .cleaning import TARGET_COL
    s = _load_all().get(dataset_id)
    t = (s.target_transform if s else "none")
    if t in ("none", "standardize"):        # affine -> R² unchanged; leave raw
        return df
    y = df[TARGET_COL].to_numpy(dtype=float)
    if t == "log1p":
        df = df.copy(); df[TARGET_COL] = np.log1p(np.clip(y, 0, None))
    elif t == "logit":
        eps = 1e-6
        p = np.clip(y, eps, 1 - eps)
        df = df.copy(); df[TARGET_COL] = np.log(p / (1 - p))
    return df


def data_present(dataset_id: str) -> bool:
    """Is the dataset's data actually available to run? (Council only targets ready datasets.)"""
    s = _load_all().get(dataset_id)
    if not s:
        return False
    if s.adapter is None:                                # built-in E. coli per-series splits
        return (ROOT / "data/splits").exists()
    if s.adapter == "yeast":
        return (ROOT / "data/extracted/seq2yield/to_import/yeast_data.csv").exists()
    local = s.source.get("local")
    if not (local and (ROOT / local).exists()):
        return False
    d = ROOT / local
    return any(d.glob(g) for g in ("*.csv*", "*.txt*", "*.tsv*"))    # csv/txt/tsv data files


def ready_ids() -> list[str]:
    return [d for d in all_ids() if data_present(d)]
