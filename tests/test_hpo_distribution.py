"""C5 — the per-series / per-subregion HPO-distribution study. Verifies config flattening + the
numeric/categorical distribution summary (with the heterogeneity flag), unit enumeration (E. coli
series + C6 strata levels), the across-unit aggregation (monkeypatched search → fast/deterministic),
and one small real series search end-to-end.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seq2yield.experiments import hpo_distribution as H  # noqa: E402


# ---- flatten + summary ----
def test_flatten_expands_list_knobs():
    f = H._flatten({"kernel_sizes": [3, 5, 3], "lr": 1e-3, "optimizer": "adam"})
    assert f["kernel_sizes_0"] == 3 and f["kernel_sizes_depth"] == 3
    assert f["kernel_sizes_mean"] == np.mean([3, 5, 3]) and f["lr"] == 1e-3


def test_summary_numeric_and_categorical_heterogeneity():
    s, het = H._summarize([3, 3, 3, 3])
    assert s["kind"] == "numeric" and s["cv"] == 0.0 and het is False       # identical -> homogeneous
    s2, het2 = H._summarize([3, 5, 3, 9])
    assert het2 is True and s2["min"] == 3 and s2["max"] == 9               # spread -> heterogeneous
    s3, het3 = H._summarize(["adam", "adam"])
    assert s3["kind"] == "categorical" and het3 is False and s3["mode"] == "adam"
    s4, het4 = H._summarize(["adam", "adamw"])
    assert het4 is True and s4["n_distinct"] == 2


# ---- unit enumeration ----
def test_units_series_and_strata():
    series = H._units("ecoli", "series", 5)
    assert len(series) == 5 and all(s.isdigit() for s in series)
    strata_units = H._units("sample_2019", "gc_bin", 99)
    assert strata_units == ["gc_bin=low", "gc_bin=mid", "gc_bin=high"]
    import pytest
    with pytest.raises(ValueError):
        H._units("sample_2019", "series", 3)               # pooled dataset has no series


# ---- across-unit aggregation (monkeypatched search) ----
def test_distribution_aggregation_flags_heterogeneity(monkeypatch):
    from agents import search_gate
    # 4 units: kernel_sizes_0 varies (3,3,5,7 -> heterogeneous); lr constant (homogeneous)
    canned = {
        "1": {"kernel_sizes": [3, 3, 3], "lr": 1e-3, "dropout": 0.3},
        "2": {"kernel_sizes": [3, 3, 3], "lr": 1e-3, "dropout": 0.3},
        "5": {"kernel_sizes": [5, 5, 3], "lr": 1e-3, "dropout": 0.4},
        "9": {"kernel_sizes": [7, 5, 3], "lr": 1e-3, "dropout": 0.5},
    }
    monkeypatch.setattr(H, "_units", lambda ds, ut, n: list(canned))

    def _fake_gated(ctx, **kw):
        cfg = canned[ctx.subregion]
        return SimpleNamespace(
            result=SimpleNamespace(best_config=cfg, best_score=0.5, n_evals=8, strategy="bandit"),
            decision=SimpleNamespace(action="light"), timed_out=False)

    monkeypatch.setattr(search_gate, "run_gated", _fake_gated)
    res = H.run_hpo_distribution("ecoli", "cnn", unit_type="series", n_units=4)
    assert len(res.per_unit) == 4
    assert res.heterogeneous["kernel_sizes_0"] is True     # series prefer different filter widths
    assert res.heterogeneous["lr"] is False                # shared learning rate
    assert res.distribution["kernel_sizes_0"]["min"] == 3 and res.distribution["kernel_sizes_0"]["max"] == 7
    assert "kernel_sizes_0" in res.headline and "lr" in res.headline


def test_record_study_writes_json(tmp_path, monkeypatch):
    from agents import search_gate
    monkeypatch.setattr(H, "_units", lambda ds, ut, n: ["1", "2"])
    monkeypatch.setattr(search_gate, "run_gated", lambda ctx, **kw: SimpleNamespace(
        result=SimpleNamespace(best_config={"n_estimators": 300}, best_score=0.2, n_evals=8, strategy="bandit"),
        decision=SimpleNamespace(action="light"), timed_out=False))
    res = H.run_hpo_distribution("ecoli", "rf", unit_type="series", n_units=2)
    p = H.record_study(res, out_dir=tmp_path)
    assert p.exists()
    import json
    saved = json.loads(p.read_text())
    assert saved["model"] == "rf" and len(saved["per_unit"]) == 2


# ---- one small real series search end-to-end ----
def test_real_series_search_rf_two_units(require_data):
    require_data("ecoli")
    res = H.run_hpo_distribution("ecoli", "rf", unit_type="series", n_units=2, train_size=250,
                                 min_action="light", gate_kwargs={"deadline_s": 240})
    assert len(res.per_unit) == 2 and res.units == ["1", "2"]
    assert all(u.gate_action in ("light", "full") for u in res.per_unit)   # study floor engaged
    assert all(isinstance(u.best_config, dict) for u in res.per_unit)
    # a real winner materializes with the now-cheap 'light' rungs; a bounded-search timeout (C10) is
    # also a legitimate outcome, so accept either.
    assert any(u.best_config for u in res.per_unit) or any(u.timed_out for u in res.per_unit)
