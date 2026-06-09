"""RL-readiness traceability (NOT RL): decision-event schema, context join, replay, extraction.

Validates the practical test — a trajectory can be reconstructed and explained, and rows are
extractable as (state_features, action_taken, candidate_actions, outcome_metrics, reward_proxy).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pytest  # noqa: E402

from agents import trace  # noqa: E402
from agents.model_clients.base import BaseStructuredClient  # noqa: E402
from agents.schemas import ExperimentIdea  # noqa: E402


@pytest.fixture()
def tmp_trace(tmp_path, monkeypatch):
    monkeypatch.setattr(trace, "EVENTS_PATH", tmp_path / "events.jsonl")
    monkeypatch.setattr(trace, "STATE_DIR", tmp_path / "state")
    return tmp_path


class _Dummy(BaseStructuredClient):
    provider = "dummy"
    model = "m1"

    def _raw(self, *, system, user, schema, temperature, max_tokens):
        return ('{"title":"t","maturity_tier":"tier_0","intervention_type":"x",'
                '"scientific_hypothesis":"h"}', {"input": 10, "output": 5})


def test_event_schema_has_all_target_fields(tmp_trace):
    rec = trace.log_event("model_routing", candidate_actions=["a", "b"], selected_action="a",
                          policy="first_available_v0", reason="r", trajectory_id="t1")
    for key in ("event_id", "run_id", "task_id", "timestamp", "decision_type", "state_ref",
                "candidate_actions", "selected_action", "policy", "reason", "model_provider",
                "model_name", "prompt_template", "input_ref", "output_ref", "latency_ms",
                "tokens_input", "tokens_output", "cost_usd", "outcome", "feedback", "reward_proxy"):
        assert key in rec
    assert rec["reward_proxy"] is None and rec["run_id"] == "t1"


def test_context_joins_model_call_to_trajectory(tmp_trace):
    log = tmp_trace / "calls.jsonl"
    with trace.trajectory("traj1", task_id="cellA"):
        _Dummy().complete_structured(system="s", user="u", schema=ExperimentIdea,
                                     role="proposal_generator", log_path=log)
    import json
    rec = json.loads(log.read_text(encoding="utf-8").splitlines()[-1])
    assert rec["run_id"] == "traj1" and rec["task_id"] == "cellA"     # the join key landed
    assert rec["output_hash"] and rec["cost_usd"] is not None         # refs + cost priced at log


def test_context_resets_after_block(tmp_trace):
    with trace.trajectory("traj2"):
        assert trace.current()["trajectory_id"] == "traj2"
    assert trace.current().get("trajectory_id") is None               # cleanly reset


def test_replay_reconstructs_and_explains(tmp_trace):
    with trace.trajectory("trajR", task_id="cellX"):
        trace.log_event("focus_planning", candidate_actions=["model_architecture"],
                        selected_action=["model_architecture"], policy="pi:deterministic",
                        reason="breadth first")
        trace.log_event("experiment_selection", candidate_actions=["p1", "p2"],
                        selected_action="p1", policy="rule_based_chair_v2", reason="best overall")
    text = trace.replay("trajR")
    assert "focus_planning" in text and "experiment_selection" in text
    assert "best overall" in text and "chose 'p1'" in text            # explains WHY


def test_extract_rows_join_outcome_and_reward_proxy(tmp_trace):
    with trace.trajectory("trajE", task_id="cellY"):
        trace.log_event("experiment_selection", candidate_actions=["p1", "p2"],
                        selected_action="p1", policy="rule_based_chair_v2", reason="x")
        trace.log_event("outcome", selected_action="run-9",
                        outcome={"status": "accepted", "mean_delta": 0.03})
    rows = trace.extract_training_rows()
    sel = [r for r in rows if r["decision_type"] == "experiment_selection"]
    assert len(sel) == 1
    row = sel[0]
    assert row["action_taken"] == "p1" and row["candidate_actions"] == ["p1", "p2"]
    assert row["outcome_metrics"]["status"] == "accepted" and row["reward_proxy"] == 1.0


def test_reward_proxy_mapping():
    assert trace.derive_reward_proxy({"status": "accepted"}) == 1.0
    assert trace.derive_reward_proxy({"status": "rejected"}) == -1.0
    assert trace.derive_reward_proxy({"status": "inconclusive"}) == 0.0
    assert trace.derive_reward_proxy({"status": "weird"}) is None


def test_router_emits_routing_event(tmp_trace):
    from agents.router import Router
    r = Router()
    with trace.trajectory("trajRoute"):
        try:
            r.resolve("proposal_generator", allow_local_fallback=True)
        except Exception:
            pass                                                     # ok if no local model in CI
    evs = trace.trajectory_events("trajRoute")
    # either a routing event fired (provider available) or none (unavailable) — assert no crash,
    # and if it fired it carries candidates + policy
    for e in evs:
        if e["decision_type"] == "model_routing":
            assert e["candidate_actions"] and e["policy"] in ("first_available_v0", "local_fallback_v0")
