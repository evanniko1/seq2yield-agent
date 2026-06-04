"""Data loading helpers for the canonical E. coli dataset and the provided split CSVs."""
from __future__ import annotations

import functools
from pathlib import Path

import pandas as pd

from .cleaning import SEQ_COL, SERIES_COL, TARGET_COL, clean_ecoli


def load_raw_ecoli(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_processed(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def build_processed_ecoli(raw_path: str | Path) -> pd.DataFrame:
    """Raw deposit CSV -> canonical cleaned frame (via the protected cleaning step)."""
    return clean_ecoli(load_raw_ecoli(raw_path))


@functools.lru_cache(maxsize=16)
def load_split_csv(path: str) -> pd.DataFrame:
    """Load and canonicalize a working/heldout split CSV (cached). Uppercases sequences."""
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")
    df[SEQ_COL] = df[SEQ_COL].astype(str).str.strip().str.upper()
    return df[df[TARGET_COL].notna()].reset_index(drop=True)


def series_subset(df: pd.DataFrame, series_id) -> pd.DataFrame:
    return df[df[SERIES_COL] == series_id]
