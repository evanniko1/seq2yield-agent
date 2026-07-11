"""G4 — per-dataset data card: provenance + distribution summary (target/GC/length/dedup/strata),
gracefully spec-only when data is absent. Synthetic frame for the compute test; real one gated."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seq2yield.data import data_card  # noqa: E402
from seq2yield.data.cleaning import SEQ_COL, TARGET_COL  # noqa: E402


def test_card_computes_distribution_and_strata(monkeypatch):
    rng = np.random.default_rng(0)
    bases = np.array(list("ACGT"))
    seqs = ["".join(rng.choice(bases, 50)) for _ in range(300)]
    df = pd.DataFrame({SEQ_COL: seqs, TARGET_COL: rng.normal(5, 1, 300)})
    monkeypatch.setattr(data_card, "_frame", lambda ds: df)
    from seq2yield.experiments import pooled_runner
    from seq2yield.data import strata
    monkeypatch.setattr(data_card.datasets, "data_present", lambda d: True)   # pass the ready gate (CI)
    monkeypatch.setattr(pooled_runner, "_frame", lambda ds: df)   # strata edges use this (offline)
    strata._edges.cache_clear()
    c = data_card.card("sample_2019")
    assert c["n"] == 300 and c["length_uniform_frac"] == 1.0
    assert "mean" in c["target"] and "skew" in c["target"] and "mean" in c["gc"]
    assert "gc_bin" in c["strata_balance"] and abs(sum(c["strata_balance"]["gc_bin"].values()) - 1) < 0.05


def test_card_is_spec_only_when_data_absent(monkeypatch):
    from seq2yield.data import datasets
    monkeypatch.setattr(datasets, "data_present", lambda d: False)
    c = data_card.card("deng_2023")
    assert c["ready"] is False and "note" in c and "n" not in c
    assert c["modality"] and c["seq_len"]                  # spec fields still present


def test_real_card(require_data):
    require_data("sample_2019")
    c = data_card.card("sample_2019")
    assert c["n"] > 1000 and 0 <= c["gc"]["mean"] <= 1 and c["duplicate_seq_frac"] >= 0
