"""R4 (run-show CLI), R6 (local-model reader), G5 (noise-ceiling framework)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


# ---- R6: local model reader ----
def test_runtime_local_model_default():
    from agents.router import runtime_local_model
    m = runtime_local_model()
    assert isinstance(m, str) and m                          # a model name (default llama3.1:8b)


# ---- G5: noise ceiling ----
def test_reliability_ceiling_perfect_and_noisy():
    from seq2yield.experiments.noise_ceiling import reliability_ceiling
    rng = np.random.default_rng(0)
    truth = rng.normal(0, 1, 500)
    perfect = reliability_ceiling(np.column_stack([truth, truth]))
    assert perfect["r2_ceiling"] > 0.99                      # identical replicates -> ceiling ~1
    noisy = reliability_ceiling(np.column_stack([truth + rng.normal(0, 1, 500),
                                                 truth + rng.normal(0, 1, 500)]))
    assert noisy["r2_ceiling"] < perfect["r2_ceiling"]       # noisy replicates -> lower ceiling


def test_noise_ceiling_unavailable_without_replicate_cols():
    from seq2yield.experiments.noise_ceiling import noise_ceiling
    c = noise_ceiling("sample_2019")                         # no replicate_cols declared
    assert c["available"] is False and "note" in c


def test_replicate_cols_field_exists():
    from seq2yield.data.datasets import DatasetSpec
    s = DatasetSpec(id="x", seq_len=10, replicate_cols=["r1", "r2"])
    assert s.replicate_cols == ["r1", "r2"]


# ---- R4: run-show CLI imports + lists ----
def test_run_show_importable():
    sys.path.insert(0, str(ROOT / "scripts"))
    import run_show
    assert hasattr(run_show, "main")
