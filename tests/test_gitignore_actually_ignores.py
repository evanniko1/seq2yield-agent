"""AG-19: the derived-data directories the repo means to ignore are actually ignored.

`.gitignore` carried `data/embeddings/*          # K2a: cached ...`. In gitignore syntax `#` starts
a comment only at the START of a line, so that trailing text was part of the PATTERN and the entry
matched nothing — leaving 512 MB of derived `.npz` permanently listed as untracked, one `git add .`
away from the history. `reports/cache/` had no entry at all.

Asserting the INTENT (these paths are ignored, those deliverables are not) rather than grepping the
file for inline `#`, because the pattern that broke was syntactically valid — it just meant
something nobody wanted. `git check-ignore` is the only authority on what git will do.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Derived or secret: regenerable caches, large artifacts, and the key file. Must be ignored.
MUST_IGNORE = [
    "data/embeddings/nt-50m/sample_2019.npz",
    "data/embeddings/some-model/whatever.npz",
    "reports/cache/families/dream2022_d2_71103_ed0fb28a3b2167e2.json",
    "reports/cache/anything/at/all.json",
    ".env",
    "data/raw/Ecoli_data.csv",
    "data/splits/whatever.json",
]

# Deliverables and the durable record. Must stay trackable — an over-broad rule is as bad as a
# broken one, and `reports/static/*` is ignored with explicit negations carved back out.
MUST_NOT_IGNORE = [
    "reports/static/2026-11-02-full56-seeds_report.md",
    "reports/static/2026-11-02-full56-seeds_curve.png",
    "reports/static/2026-07-16-council-cnn-vs-mlp-203850_loop_report.md",
    "data/manifests/datasets.json",
    "configs/baseline_registry.yaml",
    "docs/RETRACTIONS.md",
    "src/orchestration/store.py",
]


def _ignored(path: str) -> bool:
    """What git will actually do, not what the file appears to say."""
    r = subprocess.run(["git", "check-ignore", "-q", "--no-index", path],
                       cwd=ROOT, capture_output=True)
    if r.returncode not in (0, 1):
        pytest.skip(f"git check-ignore unavailable: {r.stderr.decode(errors='replace')[:120]}")
    return r.returncode == 0


@pytest.mark.parametrize("path", MUST_IGNORE)
def test_derived_and_secret_paths_are_ignored(path):
    assert _ignored(path), (
        f"{path} is NOT ignored. Check for a trailing comment on its .gitignore entry: `#` only "
        "starts a comment at the beginning of a line, so an inline one becomes part of the pattern "
        "and silently matches nothing (audit AG-19).")


@pytest.mark.parametrize("path", MUST_NOT_IGNORE)
def test_deliverables_and_source_stay_trackable(path):
    assert not _ignored(path), (
        f"{path} IS ignored, so it cannot be committed. An over-broad ignore rule loses the "
        "durable record as surely as a broken one lets 512 MB in.")


def test_no_tracked_file_is_matched_by_an_ignore_rule():
    """The condition that makes the two lists above safe to extend: if a rule ever starts matching
    something already tracked, git keeps serving it from the index and the mismatch is invisible
    until someone re-clones."""
    tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, check=True)
    r = subprocess.run(["git", "check-ignore", "--stdin"], cwd=ROOT,
                       input=tracked.stdout, capture_output=True)
    swept = [ln for ln in r.stdout.decode(errors="replace").splitlines() if ln.strip()]
    assert not swept, f"tracked files now matched by an ignore rule: {swept[:10]}"


def test_the_env_file_is_ignored_and_has_never_been_tracked():
    """The standing constraint: provider keys live in the OS keychain, and `.env` never enters the
    repo. Checked as history, not just as a rule, because a rule added later would not undo it."""
    assert _ignored(".env")
    log = subprocess.run(["git", "log", "--all", "--oneline", "--", ".env"],
                         cwd=ROOT, capture_output=True, check=True)
    assert not log.stdout.decode(errors="replace").strip(), (
        ".env appears in git history — the keys must be rotated, not just the file removed")
