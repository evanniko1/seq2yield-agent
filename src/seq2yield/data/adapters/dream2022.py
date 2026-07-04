"""Random Promoter DREAM Challenge 2022 adapter (K6) — yeast promoter -> expression.
Data: Zenodo 10.5281/zenodo.7395397 (GPRA; ~6.7M train + 71k held-out, tab-separated
sequence<TAB>expression). Provided split. Data-gated: place train/test files in the local dir.
Constructs are fixed-length in the competition format; off-length rows are dropped.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..cleaning import SEQ_COL, TARGET_COL, VALID_BASES

ROOT = Path(__file__).resolve().parents[4]


def _read(path: Path, split: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None, names=[SEQ_COL, TARGET_COL])
    df["split"] = split
    return df


def load(spec):
    local = ROOT / spec.source.get("local", f"data/extracted/{spec.id}")
    train = sorted(local.glob("*train*"))
    test = sorted(local.glob("*test*"))
    if not train:
        raise FileNotFoundError(
            f"no DREAM files under {local}. Download Zenodo {spec.source.get('zenodo')} "
            "(train/test sequence-expression tables) there (see docs/ONBOARDING.md).")
    parts = [_read(train[0], "train")] + ([_read(test[0], "test")] if test else [])
    return pd.concat(parts, ignore_index=True)


def clean(df, spec):
    out = pd.DataFrame({SEQ_COL: df[SEQ_COL].astype(str).str.strip().str.upper(),
                        TARGET_COL: pd.to_numeric(df[TARGET_COL], errors="coerce"),
                        "split": df["split"]})
    valid = ((out[SEQ_COL].str.len() == spec.seq_len)
             & out[SEQ_COL].apply(lambda s: set(s) <= VALID_BASES)
             & out[TARGET_COL].notna())
    return out[valid].reset_index(drop=True)[[SEQ_COL, TARGET_COL, "split"]]
