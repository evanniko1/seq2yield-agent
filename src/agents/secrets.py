"""BYOK secrets in the OS keychain — never .env, never the DB.

API keys (Anthropic / OpenAI / OpenRouter) live in the operating-system credential store via
`keyring`:

    * Windows  -> Credential Manager
    * macOS    -> Keychain
    * Linux    -> Secret Service (libsecret / KWallet)

Why not `.env`? AI coding agents (Claude Code, Cursor, Copilot, Codex) read project files —
a plaintext `.env` puts every secret one `Read` away from a model context window. Why not the
project SQLite store? Same reason, plus it would land in backups/exports. The keychain keeps the
secret out of the repo tree entirely and delegates at-rest encryption + per-user ACLs to the OS.

Runtime precedence (see model_clients/base.py): real environment variable > keychain > .env.
The keychain is filled into `os.environ` once, for unset keys only, at client-import time — so a
key an operator exported for a one-off run always wins over the stored one.

`keyring` is an optional dependency (`pip install -e ".[secrets]"`). Every function degrades
gracefully when it is missing or has no working backend, so the rest of the system (and CI) runs
untouched.
"""
from __future__ import annotations

import os

SERVICE = "seq2yield-agent"

# The API-key environment variables this app manages, in display order. Kept in sync with
# configs/provider_policy.yaml (api_key_env). Ollama is local and needs no key.
MANAGED_ENV_VARS = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY")


def _keyring():
    """The keyring module if importable AND backed by a real store, else None.

    A bare import can succeed on a headless box while the active backend is the `fail` or `null`
    keyring (no persistence). We treat those as unavailable so callers never think a write stuck."""
    try:
        import keyring
        from keyring.backends import fail, null
    except Exception:
        return None
    try:
        kr = keyring.get_keyring()
    except Exception:
        return None
    if isinstance(kr, (fail.Keyring, null.Keyring)):
        return None
    return keyring


def available() -> bool:
    """True when a real OS keychain backend is usable."""
    return _keyring() is not None


def backend_name() -> str:
    """Human-readable active backend (e.g. 'WinVaultKeyring'), or 'unavailable'."""
    kr = _keyring()
    if kr is None:
        return "unavailable"
    try:
        return type(kr.get_keyring()).__name__
    except Exception:
        return "unknown"


def get(name: str) -> str | None:
    """The stored secret for env-var `name`, or None (missing keyring or no entry)."""
    kr = _keyring()
    if kr is None:
        return None
    try:
        return kr.get_password(SERVICE, name)
    except Exception:
        return None


def set_secret(name: str, value: str) -> None:
    """Store `value` under env-var `name`. Raises RuntimeError if no keychain is available."""
    kr = _keyring()
    if kr is None:
        raise RuntimeError(
            "no OS keychain backend available — install `keyring` and/or a backend "
            "(Windows/macOS are built in; on Linux install libsecret)."
        )
    if not value or not value.strip():
        raise ValueError("refusing to store an empty secret")
    kr.set_password(SERVICE, name, value.strip())


def delete(name: str) -> bool:
    """Remove the stored secret for `name`. True if something was deleted, False otherwise."""
    kr = _keyring()
    if kr is None:
        return False
    try:
        kr.delete_password(SERVICE, name)
        return True
    except Exception:                      # PasswordDeleteError when the entry is absent
        return False


def load_into_env(names: tuple[str, ...] = MANAGED_ENV_VARS) -> list[str]:
    """Fill any *unset* managed env var from the keychain. Returns the names actually filled.

    Called once at client import (base.py). Real env vars are left untouched (they win)."""
    kr = _keyring()
    if kr is None:
        return []
    filled: list[str] = []
    for name in names:
        if os.environ.get(name):           # a real, non-empty env var wins
            continue
        try:
            val = kr.get_password(SERVICE, name)
        except Exception:
            val = None
        if val:
            os.environ[name] = val
            filled.append(name)
    return filled


def status(names: tuple[str, ...] = MANAGED_ENV_VARS) -> dict[str, dict]:
    """Per-key presence map for the onboarding UI. Never exposes the secret value itself."""
    out: dict[str, dict] = {}
    for name in names:
        env = os.environ.get(name) or ""
        stored = get(name) or ""
        out[name] = {
            "in_env": bool(env),
            "in_keychain": bool(stored),
            "masked": _mask(stored or env),
        }
    return out


def _mask(secret: str) -> str:
    """Show only the last 4 chars, e.g. '••••…a1b2'. Empty string for no secret."""
    s = (secret or "").strip()
    if not s:
        return ""
    if len(s) <= 4:
        return "•" * len(s)
    return "••••…" + s[-4:]
