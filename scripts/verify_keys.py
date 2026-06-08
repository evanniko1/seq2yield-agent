"""Frugal authority-provider verification (C10).

Step 1 (free): report which API keys are loaded (from .env or the environment) — names + length
only, never the secret. Step 2 (cheap): make ONE small structured call per keyed provider using
its CHEAPEST configured model (the `reviewer` model), to confirm connectivity + valid structured
output. Tiny schema, tiny max_tokens — a few cents at most.

Usage: python scripts/verify_keys.py [--call]   # --call makes the (paid) connectivity calls
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from agents.model_clients import base  # noqa: E402  (import triggers .env load)
from agents.model_clients.anthropic_client import AnthropicClient  # noqa: E402
from agents.model_clients.openai_client import OpenAIClient  # noqa: E402
from agents.schemas import ExperimentIdea  # noqa: E402

SYS = "Return ONLY JSON matching the schema."
USER = "Propose one tier_0 experiment to predict protein expression from 96nt DNA."


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--call", action="store_true", help="make the (paid) cheap connectivity calls")
    args = ap.parse_args()

    providers = yaml.safe_load((ROOT / "configs/provider_policy.yaml").read_text())["providers"]
    print("=== key presence (free) ===")
    keyed = {}
    for name in ("anthropic", "openai", "openrouter"):
        env = providers.get(name, {}).get("api_key_env", "")
        v = os.environ.get(env, "")
        print(f"  {name:11s} {env:20s} {'set (' + str(len(v)) + ' chars)' if v else 'NOT set'}")
        if v:
            keyed[name] = providers[name]["models"].get("reviewer")

    if not args.call:
        print("\n(pass --call to make one cheap structured call per keyed provider)")
        return 0

    print("\n=== cheap connectivity calls (paid; reviewer/cheapest model) ===")
    clients = {"anthropic": AnthropicClient, "openai": OpenAIClient}
    for name, model in keyed.items():
        if name not in clients:
            continue
        try:
            obj = clients[name](model).complete_structured(
                system=SYS, user=USER, schema=ExperimentIdea, role="verify",
                temperature=0.0, max_tokens=200)
            print(f"  [OK]   {name}:{model} -> tier={obj.maturity_tier} type={obj.intervention_type}")
        except Exception as e:
            print(f"  [FAIL] {name}:{model} -> {type(e).__name__}: {str(e)[:140]}")
    print("\ncall log: reports/model_calls.jsonl (tokens/cost tracked) — run scripts/show_cost.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
