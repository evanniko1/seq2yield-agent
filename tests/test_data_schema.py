"""Lock in the confirmed dataset schema from the Stage 0 audit (docs/REPRODUCTION.md §11).

If the audit manifests are absent (fresh clone without data), these tests skip rather than
fail — the data is gitignored. Run `python scripts/audit_archive.py` first to populate them.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "data/manifests/dataset_schema.json"

pytestmark = pytest.mark.skipif(
    not SCHEMA.exists(), reason="run scripts/audit_archive.py to generate dataset_schema.json"
)


def _schema():
    return json.loads(SCHEMA.read_text(encoding="utf-8"))


def _ecoli(schema):
    return next(d for d in schema["datasets"] if d["path"].endswith("Ecoli_data.csv"))


def test_ecoli_core_columns_present():
    e = _ecoli(_schema())
    assert e["sequence_columns"] == ["Sequence"]
    assert e["target_columns"] == ["Protein"]
    assert e["series_columns"] == ["mut_series"]


def test_ecoli_sequences_are_96nt():
    e = _ecoli(_schema())
    assert e["sequence_length"]["min"] == 96
    assert e["sequence_length"]["max"] == 96


def test_ecoli_row_count_matches_paper_scale():
    e = _ecoli(_schema())
    assert abs(e["n_rows"] - 228000) < 0.05 * 228000  # confirmed 227024


def test_eight_biophysical_features_present():
    e = _ecoli(_schema())
    expected = {
        "cdsCAI", "utrCdsStructureMFE", "fivepCdsStructureMFE", "threepCdsStructureMFE",
        "cdsBottleneckPosition", "cdsBottleneckRelativeStrength",
        "cdsNucleotideContentAT", "cdsHydropathyIndex",
    }
    assert expected.issubset(set(e["columns"])), expected - set(e["columns"])


def test_five_split_iterations_provided():
    schema = _schema()
    splits = schema["provided_splits"]
    assert len(splits) == 5
    for it, parts in splits.items():
        assert "working_set" in parts and "heldout_set" in parts, it
