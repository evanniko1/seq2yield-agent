"""BYOK secrets + onboarding console.

The security contract: API keys live in the OS keychain, never in .env / a config file / the DB, and
the UI never echoes a raw secret. These tests use a FAKE keyring (monkeypatched) so they pass on CI
where no real keychain backend exists — and assert graceful degradation when it is absent.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agents import secrets  # noqa: E402


class _FakeKeyring:
    """A minimal in-memory stand-in for the keyring module (get/set/delete_password)."""
    def __init__(self):
        self.store: dict[tuple[str, str], str] = {}

    def get_password(self, service, name):
        return self.store.get((service, name))

    def set_password(self, service, name, value):
        self.store[(service, name)] = value

    def delete_password(self, service, name):
        if (service, name) not in self.store:
            raise KeyError("no such entry")
        del self.store[(service, name)]


# ---------------------------------------------------------------------------- masking / privacy ---
def test_mask_reveals_only_last_four():
    assert secrets._mask("") == ""
    assert secrets._mask("ab") == "••"
    m = secrets._mask("sk-ant-abcd1234WXYZ")
    assert m.endswith("WXYZ") and m.startswith("••••") and "abcd" not in m


def test_status_never_exposes_the_raw_secret(monkeypatch):
    fake = _FakeKeyring()
    fake.set_password(secrets.SERVICE, "ANTHROPIC_API_KEY", "sk-secret-value-WXYZ")
    monkeypatch.setattr(secrets, "_keyring", lambda: fake)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    st = secrets.status(("ANTHROPIC_API_KEY",))["ANTHROPIC_API_KEY"]
    assert st["in_keychain"] is True and st["in_env"] is False
    assert "WXYZ" in st["masked"] and "secret" not in st["masked"]


# ------------------------------------------------------------------------------ precedence rule ---
def test_load_into_env_fills_unset_but_real_env_wins(monkeypatch):
    fake = _FakeKeyring()
    fake.set_password(secrets.SERVICE, "ANTHROPIC_API_KEY", "KEYCHAIN-A")
    fake.set_password(secrets.SERVICE, "OPENROUTER_API_KEY", "KEYCHAIN-O")
    monkeypatch.setattr(secrets, "_keyring", lambda: fake)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "REAL-A")          # real env var present
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)     # unset -> should be filled
    filled = secrets.load_into_env(("ANTHROPIC_API_KEY", "OPENROUTER_API_KEY"))
    assert "OPENROUTER_API_KEY" in filled and "ANTHROPIC_API_KEY" not in filled
    assert os.environ["ANTHROPIC_API_KEY"] == "REAL-A"          # real env wins over keychain
    assert os.environ["OPENROUTER_API_KEY"] == "KEYCHAIN-O"


# ------------------------------------------------------------------------ graceful degradation ---
def test_no_backend_degrades_quietly(monkeypatch):
    monkeypatch.setattr(secrets, "_keyring", lambda: None)
    assert secrets.available() is False
    assert secrets.backend_name() == "unavailable"
    assert secrets.get("ANTHROPIC_API_KEY") is None
    assert secrets.delete("ANTHROPIC_API_KEY") is False
    assert secrets.load_into_env() == []
    # set_secret must fail loudly rather than pretend it stored the key
    import pytest
    with pytest.raises(RuntimeError):
        secrets.set_secret("ANTHROPIC_API_KEY", "x")


def test_set_secret_rejects_empty(monkeypatch):
    monkeypatch.setattr(secrets, "_keyring", lambda: _FakeKeyring())
    import pytest
    with pytest.raises(ValueError):
        secrets.set_secret("ANTHROPIC_API_KEY", "   ")


# ------------------------------------------------------------------- .env -> keychain migration ---
def test_dotenv_keys_lists_only_present_managed_keys(tmp_path):
    env = tmp_path / ".env"
    env.write_text("ANTHROPIC_API_KEY=x\n# OPENAI_API_KEY=commented-out\nUNRELATED=bar\n",
                   encoding="utf-8")
    assert secrets.dotenv_keys(path=env) == ["ANTHROPIC_API_KEY"]


def test_migrate_dotenv_moves_keys_and_retires_empty_file(monkeypatch, tmp_path):
    fake = _FakeKeyring()
    monkeypatch.setattr(secrets, "_keyring", lambda: fake)
    env = tmp_path / ".env"
    env.write_text('ANTHROPIC_API_KEY=sk-ant-123\nOPENAI_API_KEY="sk-oai-456"\n', encoding="utf-8")
    rep = secrets.migrate_dotenv(path=env)
    assert set(rep["migrated"]) == {"ANTHROPIC_API_KEY", "OPENAI_API_KEY"} and not rep["failed"]
    assert rep["env_removed"] is True and not env.exists()          # only keys were in it -> removed
    assert fake.get_password(secrets.SERVICE, "ANTHROPIC_API_KEY") == "sk-ant-123"
    assert fake.get_password(secrets.SERVICE, "OPENAI_API_KEY") == "sk-oai-456"


def test_migrate_dotenv_preserves_nonkey_lines(monkeypatch, tmp_path):
    fake = _FakeKeyring()
    monkeypatch.setattr(secrets, "_keyring", lambda: fake)
    env = tmp_path / ".env"
    env.write_text("ANTHROPIC_API_KEY=sk-ant-123\nUNRELATED=keepme\n", encoding="utf-8")
    rep = secrets.migrate_dotenv(path=env)
    assert rep["migrated"] == ["ANTHROPIC_API_KEY"] and rep["env_updated"] is True and env.exists()
    txt = env.read_text(encoding="utf-8")
    assert "ANTHROPIC_API_KEY" not in txt and "UNRELATED=keepme" in txt


def test_migrate_dotenv_keeps_key_when_write_cannot_be_verified(monkeypatch, tmp_path):
    class _LossyKeyring(_FakeKeyring):
        def set_password(self, service, name, value):   # pretend to store, but nothing persists
            pass
    monkeypatch.setattr(secrets, "_keyring", lambda: _LossyKeyring())
    env = tmp_path / ".env"
    env.write_text("ANTHROPIC_API_KEY=sk-ant-123\n", encoding="utf-8")
    rep = secrets.migrate_dotenv(path=env)
    assert rep["migrated"] == [] and rep["failed"] == ["ANTHROPIC_API_KEY"]
    assert env.exists() and "ANTHROPIC_API_KEY=sk-ant-123" in env.read_text(encoding="utf-8")


# ------------------------------------------------------------------------------- onboarding app ---
def _load_onboarding():
    spec = importlib.util.spec_from_file_location("run_onboarding", ROOT / "scripts/run_onboarding.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_onboarding_page_renders_all_steps():
    m = _load_onboarding()
    r = m.app.test_client().get("/")
    assert r.status_code == 200
    html = r.data.decode("utf-8", "replace")
    for probe in ("Boot the council", "Provider mode", "OS keychain", "Local model",
                  "Test connection", "Launch council", "runtime.yaml"):
        assert probe in html, f"missing: {probe}"


def test_set_scalar_preserves_comments(tmp_path):
    m = _load_onboarding()
    p = tmp_path / "runtime.yaml"
    p.write_text("runtime:\n  mode: hybrid  # keep me\n  local_model: llama3.1:8b\n", encoding="utf-8")
    assert m._set_scalar(p, "mode", "api")
    txt = p.read_text(encoding="utf-8")
    assert "mode: api" in txt and "# keep me" in txt          # value changed, comment preserved
    assert "local_model: llama3.1:8b" in txt                  # untouched line intact
