"""C4 — best-algorithm-per-scope tournament. Verifies ranking by R², the winner-vs-rest paired
bootstrap with BH-FDR over the family, the winner-significant verdict (beats runner-up by ≥min_delta
AND survives correction), the sequence- vs series-unit fence, claim recording, and the pooled
subregion guard. The ranking/stats logic is driven by canned predictions (fast + deterministic); a
final light smoke runs a real 2-model pooled tournament.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seq2yield.data.cleaning import SEQ_COL, TARGET_COL  # noqa: E402
from seq2yield.experiments import tournament as T  # noqa: E402
from seq2yield.experiments import pooled_runner  # noqa: E402


def _canned_seq(winner_gap=0.4):
    """y_test + per-model preds where cnn ≫ rf ≫ ridge (clear, separable ranking)."""
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, 300)
    fam = {
        "cnn":   {"pred": y + rng.normal(0, 0.25, 300), "source": "biology_prior", "hparams": {"kernel_sizes": [3, 3, 3]}},
        "rf":    {"pred": y + rng.normal(0, 0.55, 300), "source": "biology_prior", "hparams": {}},
        "ridge": {"pred": y + rng.normal(0, 0.95, 300), "source": "default", "hparams": {}},
    }
    return fam, y


def _patch_pooled(monkeypatch, fam_y):
    tiny = pd.DataFrame({SEQ_COL: ["A" * 50] * 4, TARGET_COL: [1.0, 2, 3, 4]})
    monkeypatch.setattr(pooled_runner, "holdout", lambda spec, **k: (tiny, tiny))
    monkeypatch.setattr(T, "_seq_unit_family", lambda *a, **k: fam_y)


# ---- ranking + FDR + verdict (sequence unit) ----
def test_pooled_tournament_ranks_and_declares_significant_winner(monkeypatch):
    _patch_pooled(monkeypatch, _canned_seq())
    res = T.run_tournament("sample_2019", family=["cnn", "rf", "ridge"], n_boot=1000)
    assert [c.model for c in res.leaderboard] == ["cnn", "rf", "ridge"]   # ranked by R²
    assert res.winner == "cnn" and res.runner_up == "rf"
    assert res.bootstrap_unit == "sequence" and res.scope == "pooled"
    assert res.leaderboard[0].delta_vs_winner == 0.0
    assert res.winner_significant is True                 # beats rf by ≥min_delta, survives FDR
    # every non-winner carries a q-value from the family-wise correction
    assert all(c.q_value is not None for c in res.leaderboard[1:])


def test_close_family_does_not_over_declare(monkeypatch):
    # all models near-identical -> a winner exists but it is NOT significant
    rng = np.random.default_rng(1)
    y = rng.normal(0, 1, 300)
    fam = {m: {"pred": y + rng.normal(0, 0.5, 300), "source": "default", "hparams": {}}
           for m in ("cnn", "rf", "ridge")}
    _patch_pooled(monkeypatch, (fam, y))
    res = T.run_tournament("sample_2019", family=["cnn", "rf", "ridge"], n_boot=1000)
    assert res.winner_significant is False               # no runaway winner on noise-only gaps


def test_param_counts_reported_for_torch_contenders(monkeypatch):
    _patch_pooled(monkeypatch, _canned_seq())
    res = T.run_tournament("sample_2019", family=["cnn", "rf", "ridge"], n_boot=500)
    cnn = next(c for c in res.leaderboard if c.model == "cnn")
    assert cnn.n_params and cnn.n_params > 0             # capacity surfaced (C5)
    assert next(c for c in res.leaderboard if c.model == "rf").n_params is None


# ---- series unit (E. coli across series) ----
def test_series_unit_tournament_uses_series_bootstrap(monkeypatch):
    def _canned_series(*a, **k):
        rng = np.random.default_rng(0)
        base = rng.uniform(0.2, 0.6, 8)
        out = {
            "cnn":   {"per_series": base + 0.25, "source": "biology_prior", "hparams": {"kernel_sizes": [3, 3, 3]}},
            "rf":    {"per_series": base + 0.02, "source": "biology_prior", "hparams": {}},
            "ridge": {"per_series": base - 0.10, "source": "default", "hparams": {}},
        }
        return {m: {**v, "per_series": np.asarray(v["per_series"])} for m, v in out.items()}, list(range(8))

    monkeypatch.setattr(T, "_series_unit_family", _canned_series)
    res = T.run_tournament("ecoli", family=["cnn", "rf", "ridge"], n_boot=2000)
    assert res.scope == "per_series" and res.bootstrap_unit == "series" and res.n_units == 8
    assert res.winner == "cnn" and res.winner_significant is True


# ---- claim recording ----
def test_record_tournament_writes_leaderboard_and_headline_claim(monkeypatch, tmp_path):
    _patch_pooled(monkeypatch, _canned_seq())
    res = T.run_tournament("sample_2019", family=["cnn", "rf", "ridge"], n_boot=800)
    entry = T.record_tournament(res, claims_dir=tmp_path)
    assert entry["status"] == "accepted" and entry["candidate_model"] == "cnn"
    assert entry["claim"] and "best model" in entry["claim"]
    assert (tmp_path / "tournaments.jsonl").exists()
    import json
    board = json.loads((tmp_path / "tournaments.jsonl").read_text().splitlines()[-1])
    assert board["winner"] == "cnn" and len(board["leaderboard"]) == 3


# ---- guards ----
def test_malformed_subregion_is_rejected():
    # C6 implements pooled subregions, but the spec must be '<stratum>=<level>'
    import pytest
    with pytest.raises(ValueError):
        T.run_tournament("sample_2019", subregion="high_gc", family=["rf", "ridge"])


# ---- light real smoke (2 fast models, small train) ----
def test_real_pooled_smoke_two_models():
    res = T.run_tournament("sample_2019", family=["ridge", "rf"], train_size=400, n_boot=400, seed=0)
    assert res.winner in ("ridge", "rf") and res.n_units > 0
    assert res.leaderboard[0].rank == 1 and res.leaderboard[1].rank == 2
    assert res.bootstrap_unit == "sequence"
