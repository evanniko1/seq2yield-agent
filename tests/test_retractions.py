"""Guard for the retraction ledger (docs/RETRACTIONS.md).

A withdrawn result does not stay withdrawn by itself: the number sits in a dozen markdown files and
drifts back into a summary the moment someone copies a table. This test is the mechanical half of
honesty — if a retracted figure appears in a document, that document must point at the ledger.
It is deliberately dumb (string matching), because the failure mode it guards is also dumb.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

LEDGER = ROOT / "docs" / "RETRACTIONS.md"

# Figures that were published and are now retracted or superseded. Each is distinctive enough that
# a match is almost certainly a quote of the retracted result.
RETRACTED_FIGURES = [
    "0.9098",        # yeast cnn, leaky split      (R-1)
    "0.9001",        # yeast rf                    (R-1)
    "0.8963",        # yeast mlp                   (R-1)
    "0.730",         # dream2022 cnn               (R-3)
    "0.466",         # dream2022 rf                (R-3)
    "q1 0.47",       # dream2022 expression tail   (R-4)
    "0.7402",        # ecoli registry cnn @2000    (R-5)
    "0.7174",        # ecoli registry rf @2000     (R-5)
]
# Documents allowed to contain the figures without a pointer: the ledger itself, and the local
# uncommitted audit backlog (which IS the account of why they are retracted).
EXEMPT = {"docs/RETRACTIONS.md", "BACKLOG.md", "docs/BACKLOG.md"}


def _docs():
    for d in (ROOT, ROOT / "docs", ROOT / "reports" / "static"):
        for p in sorted(d.glob("*.md")):
            yield p


def test_the_ledger_exists_and_declares_its_status_vocabulary():
    text = LEDGER.read_text(encoding="utf-8")
    for word in ("RETRACTED", "SUPERSEDED", "UNAFFECTED"):
        assert word in text
    for item in ("R-1", "R-2", "R-3", "R-4", "R-5"):
        assert item in text, f"{item} missing from the ledger"
    # the two fixes that caused the retractions must be named with their commits
    assert "a13e2ba" in text and "de7a243" in text


def test_every_document_quoting_a_retracted_figure_points_at_the_ledger():
    offenders = []
    for path in _docs():
        rel = path.relative_to(ROOT).as_posix()
        if rel in EXEMPT:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        hits = [f for f in RETRACTED_FIGURES if f in text]
        if hits and "RETRACTIONS" not in text.upper():
            offenders.append((rel, hits))
    assert not offenders, ("these documents quote a retracted figure with no pointer to "
                          f"docs/RETRACTIONS.md: {offenders}")


def test_the_frozen_june_yeast_artifacts_are_not_regenerated():
    """`build_yeast.py` must never overwrite the June record: it is the only evidence of what the
    retracted claim was computed from, and re-running even the legacy protocol no longer
    reproduces it (model drift, audit ML-02)."""
    src = (ROOT / "scripts" / "build_yeast.py").read_text(encoding="utf-8")
    assert '"-legacy-rerun" if args.protocol == "legacy" else "-grouped"' in src
    assert 'default="grouped"' in src                      # the correct protocol is the default
    frozen = ROOT / "experiments" / "runs" / "yeast-baseline" / "yeast_registry.json"
    if frozen.exists():                                    # artifact present in this checkout
        doc = json.loads(frozen.read_text(encoding="utf-8"))
        assert doc["models"]["cnn"]["r2"] == 0.9098         # untouched June value
        assert "protocol" not in doc                       # not retro-stamped


def test_the_retracted_yeast_report_carries_a_banner():
    report = ROOT / "reports" / "static" / "yeast_baseline_report.md"
    if not report.exists():
        return
    head = report.read_text(encoding="utf-8")[:1200]
    assert "RETRACTED" in head and "RETRACTIONS.md" in head


def test_the_superseded_registry_report_carries_a_banner_and_the_new_one_points_back():
    """R-5: the June registry is the control half the project's claims were scored against, so the
    report must say so at the top, and the replacement must name what it replaces."""
    old = ROOT / "reports" / "static" / "2026-06-04-full56_report.md"
    if old.exists():
        head = old.read_text(encoding="utf-8")[:1400]
        assert "SUPERSEDED" in head and "RETRACTIONS.md" in head
    new = ROOT / "reports" / "static" / "2026-11-01-full56_report.md"
    if new.exists():
        head = new.read_text(encoding="utf-8")[:1400]
        assert "2026-06-04-full56" in head and "RETRACTIONS.md" in head


def test_the_canonical_registry_pointer_is_a_provenance_stamped_registry():
    """The single pointer may only name a registry the guard would accept — otherwise every run
    silently falls back to an in-run baseline and the registry is decoration (ML-02)."""
    from orchestration import baseline_guard
    run_id = baseline_guard.registry_run_id()
    assert run_id and run_id != "2026-06-04-full56", "the pointer still names the stale registry"
    status = baseline_guard.registry_status(run_id)
    if not (ROOT / "experiments" / "runs" / run_id / "metrics.csv").exists():
        return                                  # registry not present in this checkout
    assert status["ready"], status["reason"]
    verdict = baseline_guard.check(run_id, ["rf", "mlp", "cnn"])
    assert verdict["ok"], f"the canonical registry is refused by its own guard: {verdict['reason']}"


def test_new_claims_record_the_protocol_they_were_measured_under(tmp_path):
    """The ledger's whole point is that a number without its protocol is not quotable, so the
    machine-readable record carries the protocol from now on."""
    from seq2yield.experiments import claim_registry
    entry = claim_registry.record(
        run_id="t-protocol", proposal_id="p", status="accepted",
        comparison={"candidate_model": "cnn", "baseline_model": "rf", "mean_delta": 0.4,
                    "paired_bootstrap_ci": [0.34, 0.46], "ci_excludes_zero": True,
                    "bootstrap_unit": "family", "n_bootstrap_units": 2064,
                    "dataset": "dream2022", "split": {"split_strategy": "grouped_holdout"}},
        claim_allowed="cnn > rf on dream2022", claims_dir=tmp_path)
    assert entry["split_strategy"] == "grouped_holdout"
    assert entry["n_bootstrap_units"] == 2064 and entry["bootstrap_unit"] == "family"
    legacy = claim_registry.record(
        run_id="t-legacy", proposal_id="p", status="accepted",
        comparison={"mean_delta": 0.03, "bootstrap_unit": "series"},
        claim_allowed="x", claims_dir=tmp_path)
    assert legacy["split_strategy"] is None        # a pre-fix record is identifiable as such
