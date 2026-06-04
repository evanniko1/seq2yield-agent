"""Pure-logic unit tests for the audit classifiers (no data archive required)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seq2yield.data.manifests import is_junk, role_guess  # noqa: E402


def test_is_junk_filters_macos_forks():
    assert is_junk("__MACOSX/seq2yield/._Ecoli_data.csv")
    assert is_junk("seq2yield/to_import/._yeast_data.csv")
    assert is_junk("seq2yield/.DS_Store")
    assert not is_junk("seq2yield/to_import/Ecoli_data.csv")


def test_role_guess():
    assert role_guess("seq2yield/to_import/Ecoli_data.csv") == "data"
    assert role_guess("seq2yield/to_import/_saved/iteration_1/_working_set.csv") == "split"
    assert role_guess("seq2yield/to_import/_saved/iteration_1/results/deep/cnn.csv") == "result"
    assert role_guess("seq2yield/1_kmers.ipynb") == "notebook"
    assert role_guess("seq2yield/to_import/utils.py") == "script"
    assert role_guess("seq2yield/to_import/_saved/iteration_1/model.h5") == "model"
    assert role_guess("seq2yield/to_import/deeplift/deeplift.py") == "vendored"
