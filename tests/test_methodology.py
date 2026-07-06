"""Methodology fixes: the tournament's nested holdout (rank on validation, report on the untouched
test set — no selection-on-test) and the shuffled-label negative control (leakage sanity). The
canned-family path stays legacy (rank on test) for back-compat; a real small run exercises the
nested path and the control.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seq2yield.data.cleaning import SEQ_COL, TARGET_COL  # noqa: E402
from seq2yield.experiments import tournament as T  # noqa: E402
from seq2yield.experiments import controls as C  # noqa: E402
from seq2yield.experiments import pooled_runner  # noqa: E402


# ---- nested holdout ranks on val, reports on test ----
def test_nested_holdout_ranks_on_val(monkeypatch):
    # canned families with DIFFERENT val vs test orderings: B wins on val, A has higher test r2.
    def _fam(dataset, family, train_full, test, **k):
        y = test[TARGET_COL].to_numpy(dtype=float)
        rng = np.random.default_rng(0)
        out = {
            "A": {"pred": y + rng.normal(0, 0.4, len(y)), "val_r2": 0.30, "source": "s", "hparams": {}},
            "B": {"pred": y + rng.normal(0, 0.5, len(y)), "val_r2": 0.55, "source": "s", "hparams": {}},
        }
        return out, y
    tiny = pd.DataFrame({SEQ_COL: ["A" * 50] * 4, TARGET_COL: [1.0, 2, 3, 4]})
    monkeypatch.setattr(pooled_runner, "holdout", lambda spec, **k: (tiny, tiny))
    monkeypatch.setattr(T, "_seq_unit_family", _fam)
    res = T.run_tournament("sample_2019", family=["A", "B"], n_boot=300)
    assert res.selection == "nested_val" and res.winner == "B"     # chosen on val, not test
    assert res.leaderboard[0].r2_val == 0.55


def test_legacy_path_when_no_val(monkeypatch):
    # a canned family WITHOUT val_r2 falls back to ranking on the reported test r2 (back-compat)
    def _fam(dataset, family, train_full, test, **k):
        y = test[TARGET_COL].to_numpy(dtype=float)
        return ({"A": {"pred": y, "source": "s", "hparams": {}},               # perfect test
                 "B": {"pred": y + 1.0, "source": "s", "hparams": {}}}, y)
    tiny = pd.DataFrame({SEQ_COL: ["A" * 50] * 4, TARGET_COL: [1.0, 2, 3, 4]})
    monkeypatch.setattr(pooled_runner, "holdout", lambda spec, **k: (tiny, tiny))
    monkeypatch.setattr(T, "_seq_unit_family", _fam)
    res = T.run_tournament("sample_2019", family=["A", "B"], n_boot=200)
    assert res.selection == "test" and res.winner == "A"


# ---- shuffled-label negative control ----
def test_negative_control_ok_threshold():
    assert C.negative_control_ok(0.01) and C.negative_control_ok(-0.03)
    assert not C.negative_control_ok(0.20)


def test_real_shuffled_label_is_near_zero(require_data):
    require_data("sample_2019")
    r2 = C.shuffled_label_r2("sample_2019", "rf", train_size=500, seed=0)
    assert abs(r2) < 0.1 and C.negative_control_ok(r2)             # no leakage: shuffled -> ~0


def test_real_nested_tournament_has_val_and_test(require_data):
    require_data("sample_2019")
    res = T.run_tournament("sample_2019", family=["ridge", "rf"], train_size=500, n_boot=300, seed=0)
    assert res.selection == "nested_val"
    assert all(c.r2_val is not None for c in res.leaderboard)     # every contender selected on val


# ---- G3 multi-seed variance (SOTA-gap attribution) ----
def test_multiseed_structure_and_variance(monkeypatch):
    import numpy as np
    import pandas as pd
    from seq2yield.data.cleaning import SEQ_COL, TARGET_COL as TC
    frame = pd.DataFrame({SEQ_COL: ["A" * 50] * 100, TC: np.random.default_rng(0).normal(0, 1, 100)})
    monkeypatch.setattr(C, "_frames", lambda ds, sub, seed: (frame, frame))
    # seed-dependent predictions -> non-zero std
    monkeypatch.setattr(T, "_fit_predict_seq",
                        lambda ds, m, sub, test, hp, fs, sc, seed: test[TC].to_numpy()
                        + np.random.default_rng(seed).normal(0, 0.3, len(test)))
    r = C.multiseed_r2("sample_2019", "cnn", train_size=50, seeds=(0, 1, 2))
    assert len(r["r2"]) == 3 and r["std"] >= 0 and "mean" in r and r["range"] >= 0


def test_real_multiseed_rf(require_data):
    require_data("sample_2019")
    r = C.multiseed_r2("sample_2019", "rf", train_size=400, seeds=(0, 1))
    assert len(r["r2"]) == 2 and isinstance(r["mean"], float)
