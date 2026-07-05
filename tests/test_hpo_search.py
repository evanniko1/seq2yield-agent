"""C2 — the hybrid LLM-guided search engine. Verifies space sampling/perturbation stay valid, that
random search honours LLM seeds then exploits, that the successive-halving bandit promotes across
increasing rungs, that scoring uses a validation split (never the test set), and that the torch
(image) path runs under the epoch cap. Integration searches use a small synthetic training frame
(via monkeypatch) so they are fast + deterministic and never load the real 280k-row datasets.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seq2yield.data.cleaning import SEQ_COL, TARGET_COL  # noqa: E402
from seq2yield.models import registry as reg  # noqa: E402
from seq2yield.search import SearchBudget, sample_config, perturb_config, search  # noqa: E402
from seq2yield.search import engine  # noqa: E402


def _synthetic_frame(n=400, length=50, seed=0):
    rng = np.random.default_rng(seed)
    bases = np.array(list("ACGT"))
    seqs = ["".join(rng.choice(bases, length)) for _ in range(n)]
    gc = np.array([(s.count("G") + s.count("C")) / length for s in seqs])
    y = 3.0 * gc + rng.normal(0, 0.2, n)                   # learnable GC-driven signal
    return pd.DataFrame({SEQ_COL: seqs, TARGET_COL: y})


# ---- space sampling / perturbation ----
def test_sample_config_is_valid_and_bounded():
    rng = np.random.default_rng(0)
    for _ in range(20):
        c = sample_config("cnn", rng)
        assert set(c) <= set(reg.HYPERPARAMS["cnn"])       # only whitelisted knobs
        assert all(2 <= k <= 13 for k in c["kernel_sizes"])
        assert c["pool_type"] in ("max", "avg")
        rf = sample_config("rf", rng)
        assert 100 <= rf["n_estimators"] <= 1000


def test_perturb_changes_something_and_stays_valid():
    rng = np.random.default_rng(1)
    base = {"kernel_sizes": [7, 5, 3], "dropout": 0.3, "optimizer": "adam", "n_filters": [64, 128, 128]}
    changed = 0
    for _ in range(10):
        p = perturb_config("cnn", base, rng)
        assert set(p) <= set(reg.HYPERPARAMS["cnn"])
        if p != reg.clean_hyperparameters("cnn", base):
            changed += 1
    assert changed >= 5                                    # perturbation moves most of the time
    # (occasional no-ops are fine: a categorical/bool can resample to the same value)


# ---- random search: warm-start + exploitation ----
def test_random_search_honours_seeds_then_exploits(monkeypatch):
    monkeypatch.setattr(engine, "_train_frame", lambda ds, sub, seed: _synthetic_frame())
    b = SearchBudget(n_trials=6, max_train_size=300, explore_frac=0.5)
    r = search("rf", "sample_2019", strategy="random",
               seeds=[{"n_estimators": 250, "max_depth": 12}], budget=b, seed=0)
    assert r.seeds_used == 1
    assert r.trace[0]["phase"] == "seed"                   # LLM seed evaluated first
    assert {"explore", "exploit"} <= {t["phase"] for t in r.trace}
    assert r.n_evals == 6 and np.isfinite(r.best_score)
    assert r.best_config and set(r.best_config) <= set(reg.HYPERPARAMS["rf"])


def test_bandit_promotes_across_increasing_rungs(monkeypatch):
    monkeypatch.setattr(engine, "_train_frame", lambda ds, sub, seed: _synthetic_frame(600))
    b = SearchBudget(n_trials=10, halving_sizes=(150, 400), halving_keep=0.5)
    r = search("rf", "sample_2019", strategy="bandit", budget=b, seed=2)
    sizes = sorted(set(t["train_size"] for t in r.trace))
    assert sizes == [150, 400]                             # cheap rung then expensive rung
    assert r.n_evals <= 10 and np.isfinite(r.best_score)
    # more candidates at the cheap rung than the expensive one (promotion happened)
    n_cheap = sum(1 for t in r.trace if t["train_size"] == 150)
    n_exp = sum(1 for t in r.trace if t["train_size"] == 400)
    assert n_cheap > n_exp


# ---- scoring uses a validation split, never the whole train frame as its own test ----
def test_scoring_holds_out_a_validation_split(monkeypatch):
    frame = _synthetic_frame(300)
    seen = {}
    real = engine._split_train_val

    def _spy(f, val_frac, max_train_size, seed):
        tr, val = real(f, val_frac, max_train_size, seed)
        seen["tr"], seen["val"], seen["full"] = len(tr), len(val), len(f)
        return tr, val

    monkeypatch.setattr(engine, "_train_frame", lambda ds, sub, seed: frame)
    monkeypatch.setattr(engine, "_split_train_val", _spy)
    search("rf", "sample_2019", strategy="random", budget=SearchBudget(n_trials=2), seed=0)
    assert 0 < seen["val"] < seen["full"] and seen["tr"] < seen["full"]  # val carved from train


# ---- torch (image) path runs under the epoch cap ----
def test_cnn_search_runs_under_epoch_cap(monkeypatch):
    monkeypatch.setattr(engine, "_train_frame", lambda ds, sub, seed: _synthetic_frame(200))
    b = SearchBudget(n_trials=2, max_train_size=150, score_epochs=2)
    r = search("cnn", "sample_2019", strategy="random",
               seeds=[{"kernel_sizes": [3, 3, 3]}], budget=b, seed=0)
    assert r.n_evals == 2 and r.seeds_used == 1
    assert isinstance(r.best_config, dict)


# ---- guards ----
def test_unknown_model_and_strategy_raise():
    import pytest
    with pytest.raises(KeyError):
        search("nope", "sample_2019")
    with pytest.raises(ValueError):
        search("rf", "sample_2019", strategy="grid")
