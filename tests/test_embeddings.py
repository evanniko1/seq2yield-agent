"""K2a: foundation-model embedding framework — registry, cache, feature wiring, council.

No model downloads: tests write a SYNTHETIC cache and exercise the read path, so they validate
the integration (extraction itself is an offline step). `extract.py` (transformers) is never
imported here, proving the runtime/feature path is dependency-light.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agents.schemas import CouncilProposal  # noqa: E402
from seq2yield.embeddings import cache, registry  # noqa: E402
from seq2yield.features import registry as feat_registry  # noqa: E402


# ---- registry ----
def test_registry_ordered_smallest_to_largest():
    order = registry.ordered()
    orders = [registry.spec(n)["order"] for n in order]
    assert orders == sorted(orders) and order[0] == "hyenadna-tiny"


def test_registry_applicability_excludes_coding_only_for_yeast():
    yeast = registry.applicable("yeast")
    assert "utr-lm" not in yeast and "rna-fm" not in yeast and "codonbert" not in yeast  # coding-only
    assert "nt-50m" in yeast and "hyenadna-tiny" in yeast


# ---- cache roundtrip + lookup ----
def test_cache_roundtrip_and_ordered_lookup(monkeypatch, tmp_path):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    cache._load.cache_clear()
    seqs = ["AAAA", "CCCC", "GGGG"]
    vecs = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], dtype=np.float32)
    cache.write("toy", "ecoli", seqs, vecs)
    out = cache.lookup("toy", "ecoli", ["GGGG", "AAAA"])     # order preserved
    assert out.shape == (2, 2) and out[0][0] == 3.0 and out[1][0] == 1.0
    assert cache.info("toy", "ecoli") == {"model": "toy", "dataset": "ecoli", "n": 3, "dim": 2}


def test_cache_missing_file_errors_with_extract_hint(monkeypatch, tmp_path):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    cache._load.cache_clear()
    try:
        cache.lookup("nope", "ecoli", ["AAAA"])
        assert False
    except FileNotFoundError as e:
        assert "extract_embeddings.py" in str(e)


def test_cache_missing_sequence_errors(monkeypatch, tmp_path):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    cache._load.cache_clear()
    cache.write("toy", "ecoli", ["AAAA"], np.array([[1.0]], dtype=np.float32))
    try:
        cache.lookup("toy", "ecoli", ["AAAA", "TTTT"])
        assert False
    except KeyError as e:
        assert "missing" in str(e)


# ---- feature pipeline reads the cache as a flat feature_set ----
def test_embed_feature_set_builds_flat_from_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    cache._load.cache_clear()
    cache.write("toy", "ecoli", ["AAAA", "CCCC"], np.array([[1, 2], [3, 4]], dtype=np.float32))
    arr, kind = feat_registry.build("embed:toy", ["CCCC", "AAAA"], frame=None, length=96)
    assert kind == "flat" and arr.shape == (2, 2) and arr[0][0] == 3.0   # 96 -> ecoli cache


def test_embed_feature_set_length_selects_dataset(monkeypatch, tmp_path):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    cache._load.cache_clear()
    cache.write("toy", "yeast", ["AAAA"], np.array([[9.0]], dtype=np.float32))
    arr, kind = feat_registry.build("embed:toy", ["AAAA"], frame=None, length=80)  # 80 -> yeast
    assert arr[0][0] == 9.0


# ---- council schema accepts registered embeddings, rejects unknown ----
def test_proposal_accepts_registered_embedding():
    p = CouncilProposal(proposal_id="p", title="t", maturity_tier="tier_1",
                        intervention_type="feature_representation", scientific_hypothesis="h",
                        model_family="rf", comparator_model="rf", feature_set="embed:nt-50m")
    assert p.feature_set == "embed:nt-50m"


def test_proposal_rejects_unknown_embedding():
    import pytest
    with pytest.raises(Exception):
        CouncilProposal(proposal_id="p", title="t", maturity_tier="tier_1",
                        intervention_type="feature_representation", scientific_hypothesis="h",
                        model_family="rf", comparator_model="rf", feature_set="embed:does-not-exist")


def test_proposal_still_accepts_base_feature_sets():
    for fs in ("one_hot", "kmer", "mechanistic", "mixed"):
        p = CouncilProposal(proposal_id="p", title="t", maturity_tier="tier_1",
                            intervention_type="feature_representation", scientific_hypothesis="h",
                            model_family="rf", comparator_model="rf", feature_set=fs)
        assert p.feature_set == fs
