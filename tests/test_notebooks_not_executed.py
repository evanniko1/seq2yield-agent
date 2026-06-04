"""Enforce the notebook policy (docs/PROJECT_SPEC.md section 11): notebooks are forensic
seed material only and are never executed in the pipeline.

These tests are static — they never import or run a notebook.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "archive_notebooks_readonly"


def _notebooks():
    return sorted(NB_DIR.rglob("*.ipynb"))


pytestmark = pytest.mark.skipif(
    not _notebooks(), reason="run scripts/audit_archive.py to populate seed notebooks"
)


def test_notebooks_exist_in_readonly_archive():
    nbs = _notebooks()
    assert len(nbs) >= 8, f"expected the 8 project notebooks under {NB_DIR}, found {len(nbs)}"


def test_no_executed_outputs_committed():
    """Seed notebooks must carry no execution state (outputs / execution_count)."""
    offenders = []
    for nb_path in _notebooks():
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        for cell in nb.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            if cell.get("outputs"):
                offenders.append(f"{nb_path.name}: has cell outputs")
                break
            if cell.get("execution_count") not in (None, 0):
                offenders.append(f"{nb_path.name}: has execution_count")
                break
    assert not offenders, "notebooks carry execution state (must be seed-only):\n" + "\n".join(offenders)


def test_no_pipeline_source_imports_notebooks():
    """No code under src/ or scripts/ may import/execute a notebook."""
    banned = ("nbconvert", "nbclient", "papermill", "execute_notebook", "ExecutePreprocessor")
    hits = []
    for base in (ROOT / "src", ROOT / "scripts"):
        for py in base.rglob("*.py"):
            text = py.read_text(encoding="utf-8")
            for token in banned:
                if token in text:
                    hits.append(f"{py.relative_to(ROOT)} references {token}")
    assert not hits, "pipeline code must not execute notebooks:\n" + "\n".join(hits)
