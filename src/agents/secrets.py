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
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOTENV_PATH = ROOT / ".env"

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


def _clean_val(raw: str) -> str:
    """Strip whitespace and surrounding quotes from a .env value."""
    return raw.strip().strip('"').strip("'")


def dotenv_keys(path: Path = DOTENV_PATH, names: tuple[str, ...] = MANAGED_ENV_VARS) -> list[str]:
    """Managed key NAMES physically present (with a non-empty value) in the .env file. Names only —
    never returns the values. Drives the 'migrate' prompt in the onboarding UI."""
    if not path.exists():
        return []
    out: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        key, val = s.split("=", 1)
        if key.strip() in names and _clean_val(val):
            out.append(key.strip())
    return out


def migrate_dotenv(path: Path = DOTENV_PATH,
                   names: tuple[str, ...] = MANAGED_ENV_VARS) -> dict:
    """Move managed API keys from a plaintext .env into the OS keychain, then retire the plaintext.

    Safe by construction: each key is stored AND read back from the keychain before its line is
    removed from the file — a failed write never loses a key (that key stays in .env, reported under
    'failed'). After stripping the migrated lines, the file is deleted iff nothing but blanks/comments
    remains; otherwise the non-key lines are preserved. Returns a report (never the secret values):

        {available, migrated:[...], failed:[...], env_updated:bool, env_removed:bool, error?:str}
    """
    report: dict = {"available": available(), "migrated": [], "failed": [],
                    "env_updated": False, "env_removed": False}
    if not available():
        report["error"] = "no OS keychain backend available"
        return report
    if not path.exists():
        report["error"] = "no .env file found"
        return report

    lines = path.read_text(encoding="utf-8").splitlines()
    found: dict[str, tuple[int, str]] = {}          # name -> (line index, value)
    for i, line in enumerate(lines):
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        key, val = s.split("=", 1)
        key, val = key.strip(), _clean_val(val)
        if key in names and val:
            found[key] = (i, val)
    if not found:
        report["error"] = "no managed API keys found in .env"
        return report

    migrated_idx: list[int] = []
    for key, (idx, val) in found.items():
        try:
            set_secret(key, val)
            if get(key) == val:                     # verify the write stuck before we strip the line
                report["migrated"].append(key)
                migrated_idx.append(idx)
            else:
                report["failed"].append(key)
        except Exception:                           # noqa: BLE001 — any backend error -> keep in .env
            report["failed"].append(key)

    if migrated_idx:
        drop = set(migrated_idx)
        kept = [ln for i, ln in enumerate(lines) if i not in drop]
        meaningful = [ln for ln in kept
                      if ln.strip() and not ln.strip().startswith("#") and "=" in ln]
        if meaningful:
            path.write_text("\n".join(kept).rstrip("\n") + "\n", encoding="utf-8")
            report["env_updated"] = True
        else:
            path.unlink()                           # nothing but blanks/comments left -> retire it
            report["env_removed"] = True
    return report
