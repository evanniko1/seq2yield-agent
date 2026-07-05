"""Random Promoter DREAM Challenge 2022 adapter (K6) — yeast promoter -> expression.
Data: Zenodo 10.5281/zenodo.7395397. We use the 71,103-sequence MAUDE-scored file
(`filtered_test_data_with_MAUDE_expression.txt`: seq<TAB>expression, no header) as a self-contained
pooled dataset with a stratified holdout. NOTE: the 805MB `train_sequences.txt` uses a DIFFERENT
(integer bin) label scale than the MAUDE test estimates, so the two must NOT be mixed — this avoids
that pitfall (and the huge download). Constructs are fixed 110 bp; off-length rows are dropped.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..cleaning import SEQ_COL, TARGET_COL, VALID_BASES

ROOT = Path(__file__).resolve().parents[4]


def load(spec):
    local = ROOT / spec.source.get("local", f"data/extracted/{spec.id}")
    files = sorted(local.glob("*.txt")) + sorted(local.glob("*.tsv")) + sorted(local.glob("*.csv"))
    if not files:
        raise FileNotFoundError(
            f"no DREAM files under {local}. Download Zenodo {spec.source.get('zenodo')} "
            "(filtered_test_data_with_MAUDE_expression.txt) there (see docs/ONBOARDING.md).")
    return pd.concat([pd.read_csv(f, sep="\t", header=None, names=[SEQ_COL, TARGET_COL])
                      for f in files], ignore_index=True)


def clean(df, spec):
    out = pd.DataFrame({SEQ_COL: df[SEQ_COL].astype(str).str.strip().str.upper(),
                        TARGET_COL: pd.to_numeric(df[TARGET_COL], errors="coerce")})
    valid = ((out[SEQ_COL].str.len() == spec.seq_len)
             & out[SEQ_COL].apply(lambda s: set(s) <= VALID_BASES)
             & out[TARGET_COL].notna())
    return out[valid].drop_duplicates(SEQ_COL).reset_index(drop=True)[[SEQ_COL, TARGET_COL]]
