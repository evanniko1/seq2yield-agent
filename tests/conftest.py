"""Shared test fixtures. `require_data` lets the handful of real-data smoke tests SKIP gracefully
when the (gitignored, multi-GB) datasets are absent — so the full suite runs locally with data, and
CI runs every logic/agentic test without needing the data.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture
def require_data():
    from seq2yield.data import datasets

    def _req(*ids):
        missing = [d for d in ids if not datasets.data_present(d)]
        if missing:
            pytest.skip(f"needs local data (absent on CI): {missing}")
    return _req
