"""C7 — the config_transfer intervention. Verifies source-config resolution (explicit > tournament >
search), that transferring a genuinely-better config beats the target default (paired ΔR² bootstrap),
that a no-op config ties, per_series scope guarding, and claim recording. Prediction + frame seams
are monkeypatched for fast/deterministic runs, with one real explicit-config smoke.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seq2yield.data.cleaning import SEQ_COL, TARGET_COL  # noqa: E402
from seq2yield.experiments import config_transfer as C  # noqa: E402
from seq2yield.experiments import tournament as T  # noqa: E402


def _frames(n=240, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({SEQ_COL: ["A" * 50] * n, TARGET_COL: rng.normal(0, 1, n)})
    return df, df


def _patch(monkeypatch, winner_better=True):
    monkeypatch.setattr(C, "_scope_frames", lambda ds, sub, seed: _frames())

    def _fit(dataset, model, sub, test, hparams, fs, scaling, seed):
        rng = np.random.default_rng(1)
        y = test[TARGET_COL].to_numpy()
        good = hparams.get("tag") == "winner"
        noise = 0.2 if (good and winner_better) else 0.9
        return y + rng.normal(0, noise, len(y))

    monkeypatch.setattr(T, "_fit_predict_seq", _fit)


# ---- source config resolution ----
def test_find_tournament_config_reads_winner_hparams(tmp_path):
    rec = {"dataset": "yeast", "subregion": None, "winner": "cnn",
           "leaderboard": [{"model": "cnn", "hyperparameters": {"kernel_sizes": [8, 6, 4]}},
                           {"model": "rf", "hyperparameters": {"n_estimators": 500}}]}
    (tmp_path / "tournaments.jsonl").write_text(json.dumps(rec) + "\n")
    cfg = C.find_tournament_config("yeast", "cnn", None, claims_dir=tmp_path)
    assert cfg == {"kernel_sizes": [8, 6, 4]}
    assert C.find_tournament_config("yeast", "svr", None, claims_dir=tmp_path) is None


def test_resolve_prefers_explicit_then_tournament(monkeypatch, tmp_path):
    cfg, src = C.resolve_source_config("yeast", "cnn", None, config={"lr": 1e-3},
                                       feature_set="one_hot", feature_scaling="auto", seed=0)
    assert src == "explicit" and cfg == {"lr": 1e-3}
    rec = {"dataset": "yeast", "subregion": None,
           "leaderboard": [{"model": "cnn", "hyperparameters": {"kernel_sizes": [8, 6, 4]}}]}
    (tmp_path / "tournaments.jsonl").write_text(json.dumps(rec) + "\n")
    monkeypatch.setattr(C.claim_registry, "CLAIMS_DIR", tmp_path)
    cfg2, src2 = C.resolve_source_config("yeast", "cnn", None, config=None,
                                         feature_set="one_hot", feature_scaling="auto", seed=0)
    assert src2 == "tournament" and cfg2 == {"kernel_sizes": [8, 6, 4]}


# ---- transfer verdicts ----
def test_transfer_beats_default_when_config_is_better(monkeypatch):
    _patch(monkeypatch, winner_better=True)
    r = C.transfer("cnn", source_dataset="yeast", target_dataset="ecoli", target_subregion="3",
                   config={"kernel_sizes": [8, 6, 4], "tag": "winner"}, n_boot=800, record=False)
    assert r.source_of_config == "explicit"
    assert r.verdict == "beats_default" and r.excludes_zero and r.mean_delta > 0


def test_transfer_ties_when_config_is_noop(monkeypatch):
    _patch(monkeypatch, winner_better=False)      # winner config no better than default
    r = C.transfer("cnn", source_dataset="yeast", target_dataset="ecoli", target_subregion="3",
                   config={"kernel_sizes": [8, 6, 4], "tag": "winner"}, n_boot=800, record=False)
    assert r.verdict in ("ties_default", "worse_than_default")


def test_record_transfer_writes_claim(monkeypatch, tmp_path):
    _patch(monkeypatch, winner_better=True)
    r = C.transfer("cnn", source_dataset="yeast", target_dataset="ecoli", target_subregion="3",
                   config={"kernel_sizes": [8, 6, 4], "tag": "winner"}, n_boot=600, record=False)
    entry = C.record_transfer(r, claims_dir=tmp_path)
    assert entry["status"] == "accepted" and entry["claim"] and "beats" in entry["claim"]


# ---- guards ----
def test_per_series_needs_series_subregion(monkeypatch):
    with pytest.raises(ValueError):
        C._scope_frames("ecoli", None, 0)                 # per_series with no series id
    with pytest.raises(ValueError):
        C._scope_frames("ecoli", "gc_bin=high", 0)        # strata spec is not a series id


# ---- real explicit-config smoke (fast rf) ----
def test_real_transfer_smoke_rf():
    r = C.transfer("rf", source_dataset="cuperus_2017", target_dataset="sample_2019",
                   config={"n_estimators": 500, "max_depth": 12}, train_size=350, n_boot=300,
                   record=False)
    assert r.verdict in ("beats_default", "ties_default", "worse_than_default")
    assert r.n_test > 0 and isinstance(r.source_config, dict)
