"""Milestone 4: run the same prompt + schema through every available provider and validate.

Exit criterion: the same proposal prompt can be run through >=2 backends and validated against
the same schema. Providers without a key / unreachable service are reported as skipped.

Usage: python scripts/check_providers.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agents.model_clients.anthropic_client import AnthropicClient  # noqa: E402
from agents.model_clients.base import ProviderUnavailable  # noqa: E402
from agents.model_clients.ollama_client import OllamaClient  # noqa: E402
from agents.model_clients.openai_client import OpenAIClient  # noqa: E402
from agents.model_clients.openrouter_client import OpenRouterClient  # noqa: E402
from agents.schemas import ExperimentIdea  # noqa: E402

SYSTEM = ("You are a proposal generator for a bounded protein-expression ML research "
          "workflow. Propose ONE controlled Tier-0 or Tier-1 experiment. The primary metric "
          "is r2. Respond ONLY with a JSON object matching the schema.")
USER = ("Suggest one controlled experiment to improve low-data protein-expression prediction "
        "from 96nt DNA sequences, comparing against a CNN baseline on fixed per-series splits.")


def _enabled_clients():
    """Yield (label, client) for every enabled provider/model in provider_policy.yaml."""
    policy = yaml.safe_load((ROOT / "configs/provider_policy.yaml").read_text())["providers"]
    out = []
    for name, cfg in policy.items():
        if not cfg.get("enabled"):
            continue
        models = cfg.get("models", {})
        flat = []
        for v in models.values():
            flat.extend(v if isinstance(v, list) else [v])
        for model in dict.fromkeys(flat):          # dedupe, keep order
            if name == "ollama":
                out.append((f"ollama:{model}", OllamaClient(model, cfg.get("base_url", "http://localhost:11434"))))
            elif name == "anthropic":
                out.append((f"anthropic:{model}", AnthropicClient(model, cfg.get("api_key_env", "ANTHROPIC_API_KEY"))))
            elif name == "openai":
                out.append((f"openai:{model}", OpenAIClient(model, cfg.get("api_key_env", "OPENAI_API_KEY"))))
            elif name == "openrouter":
                out.append((f"openrouter:{model}", OpenRouterClient(model, cfg.get("api_key_env", "OPENROUTER_API_KEY"))))
    return out


def main() -> int:
    ok, skipped = 0, 0
    print(f"validating provider structured outputs against schema: {ExperimentIdea.__name__}\n")
    for label, client in _enabled_clients():
        try:
            obj = client.complete_structured(system=SYSTEM, user=USER, schema=ExperimentIdea,
                                             role="proposal_generator", temperature=0.3,
                                             max_tokens=800)
            print(f"  [OK]   {label}")
            print(f"         tier={obj.maturity_tier} type={obj.intervention_type}")
            print(f"         title: {obj.title}")
            ok += 1
        except ProviderUnavailable as e:
            print(f"  [SKIP] {label}: {e}")
            skipped += 1
        except Exception as e:
            print(f"  [FAIL] {label}: {type(e).__name__}: {str(e)[:160]}")
    print(f"\nvalidated: {ok}  skipped(no key/service): {skipped}")
    print(f"call log: reports/model_calls.jsonl")
    print("\nExit criterion (>=2 validated backends):",
          "MET" if ok >= 2 else f"NOT MET (only {ok}; add API keys to enable more)")
    return 0 if ok >= 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
