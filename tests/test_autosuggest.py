"""G6 — the council auto-suggests follow-on experiments to the human-accept queue (nothing runs)."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from agents.council import Council  # noqa: E402
from agents import experiment_queue  # noqa: E402


def test_autosuggest_enqueues_pending_for_ready_datasets(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment_queue, "QUEUE", tmp_path / "q.jsonl")
    monkeypatch.setattr(experiment_queue.trace, "log_event", lambda *a, **k: None)
    from seq2yield.data import datasets
    monkeypatch.setattr(datasets, "ready_ids", lambda: ["sample_2019", "cuperus_2017"])
    from seq2yield.experiments import claim_registry
    monkeypatch.setattr(claim_registry, "CLAIMS_DIR", tmp_path)   # no tournaments recorded
    out = Council(use_planner=False).autosuggest_experiments(max_suggestions=2)
    assert len(out) == 2 and all(r["status"] == "pending" and r["source"] == "council" for r in out)
    assert {r["params"]["dataset"] for r in out} == {"sample_2019", "cuperus_2017"}
    assert len(experiment_queue.list_queue("pending", tmp_path / "q.jsonl")) == 2
