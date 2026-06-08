"""Cost / token budget tracking (docs/PROJECT_SPEC §; AGENTS.md).

Aggregates the ModelCallRecord log (reports/model_calls.jsonl) into token + estimated-cost
totals (by provider / model / role), and enforces budget caps. Prices and caps are configured
in configs/experiment_budget.yaml. Local providers (Ollama) are free ($0).
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG = ROOT / "reports" / "model_calls.jsonl"
_CFG = ROOT / "configs" / "experiment_budget.yaml"

# $ per 1M tokens (input, output). PROVIDER-LEVEL fallback rates (used when a model isn't in
# DEFAULT_MODEL_PRICES). Published list prices — verify against your account/contract.
DEFAULT_PRICES = {
    "anthropic": {"input": 3.0, "output": 15.0},
    "openai": {"input": 2.0, "output": 8.0},
    "openrouter": {"input": 1.0, "output": 1.0},
    "ollama": {"input": 0.0, "output": 0.0},      # local = free
}
# PER-MODEL list prices (published rates, USD per 1M tokens). Keys match by substring so dated
# IDs resolve (e.g. "claude-haiku" matches "claude-haiku-4-5-20251001"). Longest key wins, so
# "gpt-4.1-nano" is preferred over "gpt-4.1". Authority(sonnet)/reviewer(haiku) now price apart.
DEFAULT_MODEL_PRICES = {
    "claude-opus":   {"input": 15.0, "output": 75.0},
    "claude-sonnet": {"input": 3.0,  "output": 15.0},
    "claude-haiku":  {"input": 1.0,  "output": 5.0},
    "gpt-4.1-nano":  {"input": 0.10, "output": 0.40},
    "gpt-4.1-mini":  {"input": 0.40, "output": 1.60},
    "gpt-4.1":       {"input": 2.0,  "output": 8.0},
    "gpt-4o-mini":   {"input": 0.15, "output": 0.60},
    "gpt-4o":        {"input": 2.5,  "output": 10.0},
}
DEFAULT_CAPS = {"max_total_tokens": 5_000_000, "max_total_cost_usd": 10.0, "max_calls": 5000}


def load_config() -> tuple[dict, dict, dict]:
    """Return (caps, provider_prices, model_prices). All fall back to the DEFAULT_* tables."""
    if _CFG.exists():
        cfg = yaml.safe_load(_CFG.read_text(encoding="utf-8")) or {}
        caps = {**DEFAULT_CAPS, **(cfg.get("budget") or {})}
        prices = {**DEFAULT_PRICES, **(cfg.get("prices_usd_per_million") or {})}
        model_prices = {**DEFAULT_MODEL_PRICES, **(cfg.get("model_prices_usd_per_million") or {})}
        return caps, prices, model_prices
    return dict(DEFAULT_CAPS), dict(DEFAULT_PRICES), dict(DEFAULT_MODEL_PRICES)


def _price_for(provider: str | None, model: str | None, prices: dict, model_prices: dict) -> dict:
    """Resolve the (input, output) price: most-specific model substring first, else provider."""
    model = model or ""
    for key in sorted(model_prices, key=len, reverse=True):
        if key in model:
            return model_prices[key]
    return prices.get(provider, {"input": 0.0, "output": 0.0})


def load_calls(path: str | Path = DEFAULT_LOG) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _tokens(rec: dict) -> tuple[int, int]:
    """Normalize token usage across providers (ollama input/output, OpenAI prompt/completion)."""
    u = rec.get("token_usage") or {}
    inp = u.get("input", u.get("prompt_tokens")) or 0
    out = u.get("output", u.get("completion_tokens")) or 0
    return int(inp or 0), int(out or 0)


def call_cost(rec: dict, prices: dict | None = None, model_prices: dict | None = None) -> float:
    prices = prices or DEFAULT_PRICES
    model_prices = DEFAULT_MODEL_PRICES if model_prices is None else model_prices
    inp, out = _tokens(rec)
    p = _price_for(rec.get("provider"), rec.get("model"), prices, model_prices)
    return inp / 1e6 * p.get("input", 0.0) + out / 1e6 * p.get("output", 0.0)


def summarize(records: list[dict], prices: dict | None = None,
              model_prices: dict | None = None) -> dict:
    if prices is None or model_prices is None:
        _, cfg_prices, cfg_model_prices = load_config()
        prices = prices or cfg_prices
        model_prices = cfg_model_prices if model_prices is None else model_prices
    out = {"n_calls": len(records), "n_failed": 0, "input_tokens": 0, "output_tokens": 0,
           "total_tokens": 0, "total_cost_usd": 0.0,
           "by_provider": {}, "by_model": {}, "by_role": {}}
    for r in records:
        inp, o = _tokens(r)
        cost = call_cost(r, prices, model_prices)
        out["input_tokens"] += inp
        out["output_tokens"] += o
        out["total_cost_usd"] += cost
        if not r.get("success", True):
            out["n_failed"] += 1
        for key, val in (("by_provider", r.get("provider")), ("by_model", r.get("model")),
                         ("by_role", r.get("role"))):
            d = out[key].setdefault(val, {"calls": 0, "tokens": 0, "cost_usd": 0.0})
            d["calls"] += 1
            d["tokens"] += inp + o
            d["cost_usd"] = round(d["cost_usd"] + cost, 6)
    out["total_tokens"] = out["input_tokens"] + out["output_tokens"]
    out["total_cost_usd"] = round(out["total_cost_usd"], 6)
    return out


class BudgetTracker:
    """Enforce token/cost/call caps over a set of call records (e.g. one campaign's calls)."""

    def __init__(self, caps: dict | None = None, prices: dict | None = None,
                 model_prices: dict | None = None):
        cfg_caps, cfg_prices, cfg_model_prices = load_config()
        self.caps = {**cfg_caps, **(caps or {})}
        self.prices = prices or cfg_prices
        self.model_prices = model_prices if model_prices is not None else cfg_model_prices

    def status(self, records: list[dict]) -> dict:
        s = summarize(records, self.prices, self.model_prices)
        breaches = []
        if s["total_tokens"] > self.caps["max_total_tokens"]:
            breaches.append(f"tokens {s['total_tokens']} > cap {self.caps['max_total_tokens']}")
        if s["total_cost_usd"] > self.caps["max_total_cost_usd"]:
            breaches.append(f"cost ${s['total_cost_usd']:.2f} > cap ${self.caps['max_total_cost_usd']}")
        if s["n_calls"] > self.caps["max_calls"]:
            breaches.append(f"calls {s['n_calls']} > cap {self.caps['max_calls']}")
        return {"n_calls": s["n_calls"], "total_tokens": s["total_tokens"],
                "total_cost_usd": s["total_cost_usd"], "caps": self.caps,
                "over_budget": bool(breaches), "breaches": breaches}
