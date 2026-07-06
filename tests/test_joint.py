"""C8 — joint / cross-dataset training via length-reconciliation. Verifies every strategy yields a
fixed width regardless of sequence length, that the embed strategy fails clearly without a cache,
that a pooled train-on-A→test-on-B run produces finite cross-assay metrics (per-dataset z-scored
targets), that strategies are comparable, and that deep models are rejected. Frames are
monkeypatched (synthetic, learnable) for speed; the strategy math runs for real.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seq2yield.data.cleaning import SEQ_COL, TARGET_COL  # noqa: E402
from seq2yield.experiments import joint as J  # noqa: E402


def _synth(dataset, seed=0, n=400, length=50):
    rng = np.random.default_rng(seed + len(dataset))
    bases = np.array(list("ACGT"))
    seqs = ["".join(rng.choice(bases, length)) for _ in range(n)]
    gc = np.array([(s.count("G") + s.count("C")) / length for s in seqs])
    # a SHARED signal across datasets (so cross-dataset transfer is possible) + per-dataset scale
    y = (3.0 if "cup" in dataset else 10.0) * gc + rng.normal(0, 0.3, n)
    df = pd.DataFrame({SEQ_COL: seqs, TARGET_COL: y})
    return df, df


# ---- reconciliation strategies are length-invariant in width ----
def test_every_strategy_has_fixed_width_across_lengths():
    short, long = ["ACGT" * 12], ["ACGT" * 60]
    for strat, dim in (("kmer", 256), ("pad", 5 * 300), ("adaptive_pool", 32)):
        a = J.reconcile(short, strat, max_len=300, k=4, n_bins=8)
        b = J.reconcile(long, strat, max_len=300, k=4, n_bins=8)
        assert a.shape[1] == b.shape[1] == dim


def test_kmer_is_normalized_frequency():
    x = J.reconcile(["ACGTACGT"], "kmer", k=2)
    assert x.shape[1] == 16 and abs(x.sum() - 1.0) < 1e-5


def test_embed_strategy_requires_a_cache():
    with pytest.raises(ValueError):
        J.reconcile(["ACGT"], "embed")                    # no dataset/embed_model
    with pytest.raises(ValueError):
        J.reconcile(["ACGT"], "not_a_strategy")


# ---- joint train-on-A -> test-on-B ----
def test_joint_trains_across_datasets_and_predicts_target(monkeypatch):
    monkeypatch.setattr(J, "_dataset_frames", lambda ds, seed: _synth(ds, seed))
    r = J.run_joint(["cuperus_2017"], "sample_2019", model="rf", strategy="kmer",
                    train_size_per=300, seed=0)
    assert r.feature_dim == 256 and r.n_train > 0 and r.n_test > 0
    assert np.isfinite(r.spearman) and r.spearman > 0.3   # shared GC signal transfers across scale
    assert r.metric_primary == "spearman"


def test_multi_source_pool(monkeypatch):
    monkeypatch.setattr(J, "_dataset_frames", lambda ds, seed: _synth(ds, seed))
    r = J.run_joint(["cuperus_2017", "yeast"], "sample_2019", model="ridge",
                    strategy="adaptive_pool", train_size_per=200, seed=1)
    assert r.n_train >= 300 and r.feature_dim == 32 and np.isfinite(r.spearman)


def test_compare_strategies_returns_one_per_strategy(monkeypatch):
    monkeypatch.setattr(J, "_dataset_frames", lambda ds, seed: _synth(ds, seed))
    out = J.compare_strategies(["cuperus_2017"], "sample_2019", model="rf",
                               strategies=("kmer", "adaptive_pool", "embed"), train_size_per=200)
    assert [o.strategy for o in out] == ["kmer", "adaptive_pool", "embed"]
    assert np.isfinite(out[0].spearman)                   # kmer works
    assert out[2].metric_primary.startswith("error")      # embed has no cache -> graceful


# ---- guards ----
def test_deep_models_are_rejected(monkeypatch):
    monkeypatch.setattr(J, "_dataset_frames", lambda ds, seed: _synth(ds, seed))
    with pytest.raises(ValueError):
        J.run_joint(["cuperus_2017"], "sample_2019", model="cnn")
