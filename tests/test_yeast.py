"""Tests for the yeast benchmark: cleaning, stratified holdout, sequence-level bootstrap."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from seq2yield.data.cleaning import SEQ_COL, SERIES_COL, TARGET_COL, clean_yeast  # noqa: E402
from seq2yield.statistics.bootstrap import bootstrap_r2_ci, paired_bootstrap_r2  # noqa: E402


def test_clean_yeast_renames_and_filters():
    df = pd.DataFrame({
        "Unnamed: 0": [0, 1, 2],
        "native_gene": ["YOL116W", "YOL116W", "YBR1"],
        "sequence": ["acgt" * 20, "ACGT" * 20, "xy"],   # 80nt lower, 80nt upper, invalid
        "protein": [73.2, 58.0, 10.0],
    })
    out = clean_yeast(df)
    assert list(out.columns) == [SEQ_COL, TARGET_COL, SERIES_COL]
    assert len(out) == 2 and out[SEQ_COL].str.isupper().all()
    assert (out[SEQ_COL].str.len() == 80).all()


def test_stratified_holdout_covers_every_gene():
    import build_yeast
    rng = np.random.default_rng(0)
    rows = []
    for g in range(20):
        for _ in range(20):
            rows.append({SEQ_COL: "".join(rng.choice(list("ACGT"), 80)),
                         TARGET_COL: rng.uniform(0, 100), SERIES_COL: g})
    df = pd.DataFrame(rows)
    train, test = build_yeast.stratified_holdout(df, frac=0.1, seed=1)
    assert len(train) + len(test) == len(df)
    assert set(test[SERIES_COL]) == set(range(20))     # every gene represented in test


def test_sequence_bootstrap_detects_difference():
    rng = np.random.default_rng(0)
    y = rng.uniform(0, 100, 300)
    good = y + rng.normal(0, 3, 300)        # strong predictor
    bad = y + rng.normal(0, 25, 300)        # weak predictor
    ci = bootstrap_r2_ci(y, good, seed=0)
    assert ci["ci"][0] <= ci["r2"] <= ci["ci"][1] + 1e-9
    pb = paired_bootstrap_r2(y, good, bad, seed=0)
    assert pb["mean_delta"] > 0 and pb["excludes_zero"]
