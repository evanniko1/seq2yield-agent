"""K6: dataset onboarding — DatasetSpec registry, data-presence gating, adapter cleaning,
per-dataset applicability, and the intake-audit gate. No external data required (synthetic frames).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agents import question_space as qs  # noqa: E402
from seq2yield.data import datasets  # noqa: E402
from seq2yield.data.adapters import sample_2019 as s19  # noqa: E402
from seq2yield.data.cleaning import SEQ_COL, TARGET_COL  # noqa: E402


# ---- registry ----
def test_registry_loads_three_datasets_with_specs():
    ids = set(datasets.all_ids())
    assert {"ecoli", "yeast", "sample_2019"} <= ids
    assert datasets.seq_len("ecoli") == 96 and datasets.seq_len("yeast") == 80
    assert datasets.seq_len("sample_2019") == 50
    assert datasets.spec("sample_2019").structure == "pooled"
    assert datasets.spec("ecoli").structure == "per_series"


def test_applicable_feature_sets_excludes_mechanistic_for_utr():
    assert "mechanistic" in datasets.applicable_feature_sets("ecoli")
    assert "mechanistic" not in datasets.applicable_feature_sets("sample_2019")


# ---- data-presence gating ----
def test_ready_ids_excludes_datasets_without_data():
    ready = datasets.ready_ids()
    assert "ecoli" in ready and "yeast" in ready          # data present locally
    assert "sample_2019" not in ready                     # GEO data not downloaded


def test_question_space_only_enumerates_ready_datasets():
    ds = {c.dataset for c in qs.enumerate_cells()}
    assert ds == {"ecoli", "yeast"}                       # sample_2019 gated out (no data)
    # per-dataset applicability: yeast feature cells never use mechanistic
    yeast_fs = {c.feature_set for c in qs.enumerate_cells()
                if c.dataset == "yeast" and c.intervention_type == "feature_representation"}
    assert "mechanistic" not in yeast_fs and "mixed" not in yeast_fs


# ---- sample_2019 adapter cleaning (synthetic GEO-like frame) ----
def _synthetic_sample(n=200, seed=0):
    rng = np.random.default_rng(seed)
    bases = np.array(list("ACGT"))
    seqs = ["".join(rng.choice(bases, 50)) for _ in range(n)]
    seqs += ["ACGT" * 10]                                  # a 40nt (wrong length) -> must be dropped
    rl = list(rng.normal(5, 1, n)) + [3.0]
    reads = list(rng.integers(10, 1000, n)) + [5]
    return pd.DataFrame({"utr": seqs, "rl": rl, "total_reads": reads})


def test_sample_2019_clean_yields_canonical_frame():
    spec = datasets.spec("sample_2019")
    out = s19.clean(_synthetic_sample(), spec)
    assert list(out.columns) == [SEQ_COL, TARGET_COL, "split"]
    assert (out[SEQ_COL].str.len() == 50).all()           # wrong-length row dropped
    assert set(out["split"]) <= {"train", "test"} and "test" in set(out["split"])


def test_sample_2019_clean_has_no_leakage_between_splits():
    out = s19.clean(_synthetic_sample(300), datasets.spec("sample_2019"))
    tr = set(out[out["split"] == "train"][SEQ_COL])
    te = set(out[out["split"] == "test"][SEQ_COL])
    assert tr.isdisjoint(te) or len(tr & te) == 0         # provided split is leak-free


# ---- intake audit on a registered-but-synthetic dataset ----
def test_intake_audit_passes_clean_dataset(monkeypatch, tmp_path):
    import importlib
    onboard = importlib.import_module("onboard_dataset") if "onboard_dataset" in sys.modules else None
    from seq2yield.data import adapters
    # audit reads the cleaned frame via adapters.frame_for; stub it with a clean synthetic frame
    clean = s19.clean(_synthetic_sample(20000), datasets.spec("sample_2019"))
    monkeypatch.setattr(datasets, "data_present", lambda d: True)
    monkeypatch.setattr(adapters, "frame_for", lambda d: clean)
    sys.path.insert(0, str(ROOT / "scripts"))
    import onboard_dataset
    res = onboard_dataset.audit("sample_2019")
    assert res["pass"] is True
    assert res["checks"]["length_uniform"]["pass"] and res["checks"]["throughput_floor"]["pass"]
    assert res["checks"]["no_train_test_leakage"]["pass"]
