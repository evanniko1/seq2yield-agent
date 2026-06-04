"""Tests for one-hot DNA encoding and the cleaning step."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seq2yield.data.cleaning import clean_ecoli  # noqa: E402
from seq2yield.features.one_hot import one_hot, one_hot_flat  # noqa: E402


def test_one_hot_shape_and_values():
    arr = one_hot(["ACGT"], length=4)
    assert arr.shape == (1, 4, 4)
    # A->0, C->1, G->2, T->3 on the diagonal
    assert arr[0, 0, 0] == 1.0 and arr[0, 1, 1] == 1.0
    assert arr[0, 2, 2] == 1.0 and arr[0, 3, 3] == 1.0
    assert arr.sum() == 4.0  # exactly one hot per position


def test_one_hot_flat_shape():
    flat = one_hot_flat(["ACGT", "TTTT"], length=4)
    assert flat.shape == (2, 16)


def test_clean_ecoli_uppercases_and_filters():
    df = pd.DataFrame({
        "Unnamed: 0": [0, 1, 2],
        "Sequence": ["acgt" * 24, "ACGT" * 24, "xy"],  # 96nt lower, 96nt upper, invalid
        "Protein": [10.0, 20.0, 30.0],
        "mut_series": [1, 1, 2],
    })
    out = clean_ecoli(df)
    assert "Unnamed: 0" not in out.columns
    assert len(out) == 2  # invalid short/non-ACGT row dropped
    assert out["Sequence"].str.isupper().all()
    assert (out["Sequence"].str.len() == 96).all()
