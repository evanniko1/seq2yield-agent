"""Role -> model client resolution (docs/AGENTS.md §5, configs/provider_policy.yaml).

Maps a role to its provider_class (agent_roles.yaml), then to an ordered provider list
(provider_policy.yaml provider_class_map), and returns the first enabled + available client.
Authority roles must use a direct provider.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from .model_clients.anthropic_client import AnthropicClient
from .model_clients.base import ModelClient, ProviderUnavailable
from .model_clients.ollama_client import OllamaClient
from .model_clients.openai_client import OpenAIClient
from .model_clients.openrouter_client import OpenRouterClient

ROOT = Path(__file__).resolve().parents[2]
_DIRECT = {"anthropic", "openai"}


def _load(name: str) -> dict:
    return yaml.safe_load((ROOT / "configs" / name).read_text(encoding="utf-8"))


def _model_for(provider_cfg: dict, provider_class: str):
    """Pick the model name for a provider given the role's provider_class."""
    models = provider_cfg.get("models", {})
    key = "authority" if provider_class == "authority" else "diversity"
    val = models.get(key) or next(iter(models.values()), None)
    if isinstance(val, list):
        return val[0] if val else None
    return val


def _build(provider: str, model: str, provider_cfg: dict) -> ModelClient:
    if provider == "ollama":
        return OllamaClient(model, base_url=provider_cfg.get("base_url", "http://localhost:11434"))
    if provider == "anthropic":
        return AnthropicClient(model, api_key_env=provider_cfg.get("api_key_env", "ANTHROPIC_API_KEY"))
    if provider == "openai":
        return OpenAIClient(model, api_key_env=provider_cfg.get("api_key_env", "OPENAI_API_KEY"))
    if provider == "openrouter":
        return OpenRouterClient(model, api_key_env=provider_cfg.get("api_key_env", "OPENROUTER_API_KEY"))
    raise ValueError(f"unknown provider '{provider}'")


def _available(provider: str, client: ModelClient) -> bool:
    if provider == "ollama":
        return OllamaClient.available(getattr(client, "base_url", "http://localhost:11434"))
    return bool(getattr(client, "available", lambda: True)())


class Router:
    def __init__(self):
        self.policy = _load("provider_policy.yaml")
        self.roles = _load("agent_roles.yaml")["agent_roles"]

    def provider_class(self, role: str) -> str:
        return self.roles.get(role, {}).get("provider_class", "diversity")

    def candidates(self, role: str) -> list[str]:
        pclass = self.provider_class(role)
        order = self.policy["provider_class_map"].get(pclass, [])
        # authority roles: keep only direct providers
        rp = self.policy.get("role_policy", {}).get(role, {})
        if rp.get("require_direct_provider"):
            order = [p for p in order if p in _DIRECT]
            if rp.get("allowed_providers"):
                order = [p for p in order if p in rp["allowed_providers"]]
        return order

    def _emit_routing(self, role, candidates, selected, policy, reason):
        try:                                             # RL-trace (best-effort, never blocks)
            from . import trace
            trace.log_event("model_routing", candidate_actions=candidates,
                            selected_action=selected, policy=policy, reason=reason,
                            state={"role": role})
        except Exception:
            pass

    def resolve(self, role: str, *, require_available: bool = True,
                allow_local_fallback: bool = False) -> ModelClient:
        providers = self.policy["providers"]
        pclass = self.provider_class(role)
        candidates = self.candidates(role)
        errs = []
        for provider in candidates:
            pcfg = providers.get(provider, {})
            if not pcfg.get("enabled"):
                errs.append(f"{provider}: disabled")
                continue
            model = _model_for(pcfg, pclass)
            if not model:
                errs.append(f"{provider}: no model configured")
                continue
            client = _build(provider, model, pcfg)
            if require_available and not _available(provider, client):
                errs.append(f"{provider}: unavailable (key/service)")
                continue
            self._emit_routing(role, candidates, f"{provider}:{model}", "first_available_v0",
                               f"first enabled+available of {pclass} class; skipped: {errs}")
            return client

        # Offline/keyless DEV fallback: let an authority role borrow a local model when no
        # direct provider is available. Loudly marked; never the default (AGENTS.md §5).
        if allow_local_fallback:
            for provider in self.policy["provider_class_map"].get("diversity", []):
                pcfg = providers.get(provider, {})
                if not pcfg.get("enabled"):
                    continue
                model = _model_for(pcfg, "diversity")
                client = _build(provider, model, pcfg)
                if _available(provider, client):
                    client.local_fallback_for = role          # type: ignore[attr-defined]
                    self._emit_routing(role, candidates, f"{provider}:{model}",
                                       "local_fallback_v0",
                                       f"no direct provider available; DEV local fallback for {role}")
                    return client
        raise ProviderUnavailable(f"no provider available for role '{role}': {errs}")
