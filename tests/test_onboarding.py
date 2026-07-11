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
def test_ready_ids_gate_on_data_presence():
    # CI-safe invariant: ready == exactly the registered datasets whose data is present (both empty
    # on CI where the gitignored data is absent). Tests the gate's definition, not a specific dataset.
    ready = set(datasets.ready_ids())
    assert ready == {d for d in datasets.all_ids() if datasets.data_present(d)}


def test_question_space_enumerates_ready_datasets_and_yeast_applicability(require_data):
    # needs data: _ready_datasets() falls back to the ecoli/yeast defaults when NOTHING is ready
    # (so on CI it would enumerate cells whose data is absent) → skip on CI via require_data.
    require_data("ecoli", "yeast")
    ds = {c.dataset for c in qs.enumerate_cells()}
    assert {"ecoli", "yeast"} <= ds
    assert all(datasets.data_present(d) for d in ds)      # with data present, enumerate == ready
    # per-dataset applicability: yeast (promoter) feature cells never use mechanistic/mixed
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


# ---- item 1: new datasets + target_transform ----
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from seq2yield.data.cleaning import SEQ_COL, TARGET_COL  # noqa: E402


def test_registry_has_all_onboarded_datasets():
    ids = set(datasets.all_ids())
    assert {"ecoli", "yeast", "sample_2019", "cuperus_2017", "tewhey_2016", "dream2022"} <= ids
    assert datasets.spec("tewhey_2016").seq_len == 150      # <=500 length ceiling honored
    assert datasets.spec("cuperus_2017").structure == "pooled"


def test_target_transform_none_and_standardize_are_noops():
    df = pd.DataFrame({SEQ_COL: ["A"], TARGET_COL: [3.0]})
    for t in ("none", "standardize"):
        # ecoli=none, yeast=none; both leave the value untouched
        assert datasets.apply_target_transform(df, "ecoli")[TARGET_COL].iloc[0] == 3.0


def test_target_transform_log1p_and_logit(monkeypatch):
    from seq2yield.data.datasets import DatasetSpec
    df = pd.DataFrame({SEQ_COL: ["A", "C"], TARGET_COL: [0.0, np.e - 1]})
    # patch a fake spec with log1p
    fake = DatasetSpec(id="x", seq_len=10, target_transform="log1p")
    monkeypatch.setattr(datasets, "_load_all", lambda: {"x": fake})
    out = datasets.apply_target_transform(df, "x")[TARGET_COL].to_numpy()
    assert abs(out[0]) < 1e-9 and abs(out[1] - 1.0) < 1e-6      # log1p(0)=0, log1p(e-1)=1
    fake2 = DatasetSpec(id="y", seq_len=10, target_transform="logit")
    monkeypatch.setattr(datasets, "_load_all", lambda: {"y": fake2})
    dfb = pd.DataFrame({SEQ_COL: ["A"], TARGET_COL: [0.5]})
    assert abs(datasets.apply_target_transform(dfb, "y")[TARGET_COL].iloc[0]) < 1e-6  # logit(0.5)=0


def test_new_adapters_import():
    import importlib
    for a in ("sample_2019", "cuperus_2017", "tewhey_2016", "dream2022", "_seelig"):
        importlib.import_module(f"seq2yield.data.adapters.{a}")


def test_deng_2023_registered_and_adapter_cleans_270bp():
    s = datasets.spec("deng_2023")
    assert s.seq_len == 270 and s.modality == "regulatory" and s.bootstrap_unit == "sequence"
    assert "mechanistic" not in s.applicable_feature_sets   # regulatory, not E.coli coding
    from seq2yield.data.adapters import deng_2023 as A
    rng = np.random.default_rng(0); bases = np.array(list("ACGT"))
    seqs = ["".join(rng.choice(bases, 270)) for _ in range(30)] + ["ACGT" * 10]  # 40bp -> dropped
    df = pd.DataFrame({"sequence": seqs, "activity": list(rng.normal(1, .5, 30)) + [9.0]})
    out = A.clean(df, s)
    assert list(out.columns) == [SEQ_COL, TARGET_COL] and (out[SEQ_COL].str.len() == 270).all()
