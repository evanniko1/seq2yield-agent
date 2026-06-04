"""Provider layer tests: structured-output validate/retry/log, router resolution, live Ollama."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agents.model_clients.base import BaseStructuredClient, ProviderUnavailable  # noqa: E402
from agents.model_clients.ollama_client import OllamaClient  # noqa: E402
from agents.router import Router  # noqa: E402
from agents.schemas import ExperimentIdea  # noqa: E402

GOOD = ('{"title":"t","maturity_tier":"tier_0","intervention_type":"x",'
        '"scientific_hypothesis":"h","primary_metric":"r2"}')


class _Fake(BaseStructuredClient):
    provider = "fake"
    model = "m"

    def __init__(self, outputs):
        self._outputs = list(outputs)
        self.calls = 0

    def _raw(self, **kw):
        out = self._outputs[min(self.calls, len(self._outputs) - 1)]
        self.calls += 1
        if isinstance(out, Exception):
            raise out
        return out, {"input": 1, "output": 1}


def test_structured_success_and_log(tmp_path):
    c = _Fake([GOOD])
    obj = c.complete_structured(system="s", user="u", schema=ExperimentIdea,
                                log_path=tmp_path / "calls.jsonl")
    assert isinstance(obj, ExperimentIdea) and obj.maturity_tier == "tier_0"
    assert (tmp_path / "calls.jsonl").read_text().strip()


def test_retry_then_succeed(tmp_path):
    c = _Fake(["not json", GOOD])
    obj = c.complete_structured(system="s", user="u", schema=ExperimentIdea,
                                log_path=tmp_path / "c.jsonl")
    assert obj.title == "t" and c.calls == 2


def test_provider_unavailable_raises(tmp_path):
    c = _Fake([ProviderUnavailable("no key")])
    with pytest.raises(ProviderUnavailable):
        c.complete_structured(system="s", user="u", schema=ExperimentIdea,
                              log_path=tmp_path / "c.jsonl")


def test_router_authority_is_direct_only():
    r = Router()
    assert r.provider_class("chair") == "authority"
    cands = r.candidates("chair")
    assert "ollama" not in cands and set(cands) <= {"anthropic", "openai"}


def test_router_diversity_includes_local():
    r = Router()
    assert r.provider_class("proposal_generator") == "diversity"
    assert "ollama" in r.candidates("proposal_generator")


@pytest.mark.skipif(not OllamaClient.available(), reason="Ollama not running")
def test_live_ollama_structured(tmp_path):
    models = ["qwen2.5-coder:14b", "llama3.2:latest"]
    c = OllamaClient(models[0])
    obj = c.complete_structured(
        system="Return ONLY JSON matching the schema.",
        user="Propose one tier_0 experiment to predict protein expression from DNA.",
        schema=ExperimentIdea, role="proposal_generator", max_tokens=400,
        log_path=tmp_path / "c.jsonl")
    assert obj.maturity_tier in {"tier_0", "tier_1", "tier_2", "tier_3"}
