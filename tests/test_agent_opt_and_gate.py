"""Agent-optimization (prompt caching + concurrent reviewers, thread-safe logging) and the (b)
blocking-flag hard gate. All offline / data-free (fakes for the provider + council)."""
from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


# ---------------------------------------------------------------------- (a) prompt caching ---
def test_anthropic_marks_cache_breakpoints_and_reports_cache_usage(monkeypatch):
    from agents.model_clients.anthropic_client import AnthropicClient

    captured: dict = {}

    class _Usage:
        input_tokens, output_tokens = 12, 5
        cache_read_input_tokens, cache_creation_input_tokens = 9, 0

    class _Block:
        type, name, input = "tool_use", "emit_result", {"x": 1}

    class _Resp:
        usage = _Usage()
        content = [_Block()]

    class _FakeClient:
        class messages:
            @staticmethod
            def create(**kw):
                captured.update(kw)
                return _Resp()

    class S(BaseModel):
        x: int

    c = AnthropicClient("claude-x")
    c.api_key, c._client = "sk-test", _FakeClient()
    out, usage = c._raw(system="ROLE PERSONA", user="hi", schema=S, temperature=0.0, max_tokens=50)

    assert out == {"x": 1}
    assert usage["cache_read"] == 9 and usage["cache_write"] == 0        # surfaced for telemetry
    assert isinstance(captured["system"], list)                          # system -> cached block
    assert captured["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert captured["tools"][0]["cache_control"] == {"type": "ephemeral"}   # tool schema cached too


def test_anthropic_cache_can_be_disabled(monkeypatch):
    from agents.model_clients.anthropic_client import AnthropicClient

    captured: dict = {}

    class _Resp:
        class usage:
            input_tokens, output_tokens = 1, 1
            cache_read_input_tokens = cache_creation_input_tokens = 0
        content = [type("B", (), {"type": "tool_use", "name": "emit_result", "input": {"x": 1}})()]

    class _FakeClient:
        class messages:
            @staticmethod
            def create(**kw):
                captured.update(kw)
                return _Resp()

    class S(BaseModel):
        x: int

    c = AnthropicClient("claude-x")
    c.api_key, c._client = "sk-test", _FakeClient()
    c._raw(system="P", user="hi", schema=S, temperature=0.0, max_tokens=5, cache=False)
    assert captured["system"] == "P"                                     # left as a plain string
    assert "cache_control" not in captured["tools"][0]


# ---------------------------------------------------------- (a) thread-safe logging ---
def test_log_call_is_thread_safe(tmp_path):
    from agents.model_clients import base
    p = tmp_path / "calls.jsonl"

    def _w():
        rec = base.ModelCallRecord(provider="x", model="m", role="r", prompt_hash="h",
                                   schema_name="S")
        base.log_call(rec, p)

    threads = [threading.Thread(target=_w) for _ in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    lines = p.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 50
    for ln in lines:
        json.loads(ln)                                                   # no interleaved partials


# ---------------------------------------------------- (a) concurrent reviewer fan-out ---
def test_review_proposal_preserves_order_concurrently(monkeypatch):
    import agents.council as C

    # roster-independent: fan out over a fixed multi-reviewer list so concurrency is exercised
    # regardless of the live roster (post-collapse the live roster is a single critic).
    reviewers = ["modeling_reviewer", "biology_reviewer", "doe_strategist"]
    monkeypatch.setattr(C.roles, "reviewers", lambda: reviewers)
    monkeypatch.setattr(C.prompting, "reviewer_prompt",
                        lambda r, p, peer_summary="": ("sys", "usr"))

    council = C.Council(allow_local_fallback=True, use_planner=False)
    council.review_workers = 4

    def _fake_ask(role, prompt, schema, **kw):
        return C.CouncilReviewItem(role=role, score_feasibility=4, score_scientific_value=4,
                                   score_confoundedness=4, score_reproducibility=4), "fake:model"

    monkeypatch.setattr(council, "_ask", _fake_ask)

    class _P:
        proposal_id = "p1"

        def model_dump(self):
            return {}

    items = council._review_proposal(_P(), "")
    assert [i.role for i in items] == reviewers                          # order despite threads
    council.review_workers = 1                                           # serial path, same result
    assert [i.role for i in council._review_proposal(_P(), "")] == reviewers


# ------------------------------------------------------------ (b) blocking-flag hard gate ---
def test_critic_carries_blocking_on_leakage():
    from seq2yield.diagnostics.critic import evaluate
    flags = evaluate({"leakage": {"leak_frac": 0.2}})
    leak = [f for f in flags if f["id"] == "train_test_leakage"]
    assert leak and leak[0]["blocking"] is True


def test_flag_gate_withholds_accepted_result_on_blocking_flag(tmp_path):
    from orchestration import execution_harness as H
    cmp = {"status": "accepted",
           "methodology_flags": [{"id": "train_test_leakage", "severity": "high", "blocking": True}]}
    H._flag_gate(cmp, tmp_path)
    assert cmp["status"] == "inconclusive"
    assert cmp["gated_by_flags"] == ["train_test_leakage"]
    assert any("withheld" in r for r in cmp["reasons"])


def test_flag_gate_is_conservative(tmp_path):
    from orchestration import execution_harness as H
    # advisory (non-blocking) flag never gates
    a = {"status": "accepted",
         "methodology_flags": [{"id": "overfit", "severity": "medium", "blocking": False}]}
    H._flag_gate(a, tmp_path)
    assert a["status"] == "accepted"
    # a blocking flag never *upgrades* a non-accepted verdict
    r = {"status": "rejected",
         "methodology_flags": [{"id": "train_test_leakage", "blocking": True}]}
    H._flag_gate(r, tmp_path)
    assert r["status"] == "rejected"
