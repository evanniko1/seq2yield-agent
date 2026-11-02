"""Guards for AG-04 / IM-02: the local consoles reject forged cross-origin writes.

`127.0.0.1` is not a security boundary. A page the user merely visits can POST a plain form to a
localhost port — form posts are CORS-simple, so the browser does not block them and nothing in
these apps rejected them. The audit demonstrated a cross-origin POST returning **200** while
performing five config writes (`unlocked_tier=tier_3`, `max_total_cost_usd=9999`, …), spawning a
paid `run_council.py`, and calling `secrets.delete('ANTHROPIC_API_KEY')`.

Each test here is the attack, asserted to fail, plus the legitimate request asserted to succeed —
because a guard that also breaks the console is not a fix.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orchestration import console_guard as G  # noqa: E402

EVIL = "https://evil.example"


@pytest.fixture
def app(monkeypatch, tmp_path):
    """The operator console with its config files redirected into tmp_path."""
    import importlib
    mod = importlib.import_module("importlib.util")
    spec = mod.spec_from_file_location("config_app", ROOT / "scripts" / "config_app.py")
    m = mod.module_from_spec(spec)
    sys.modules["config_app"] = m
    spec.loader.exec_module(m)
    for name, text in (("POLICY", "selection_bonuses:\n  novel_dataset: 1.0   # keep me\n"),
                       ("BUDGET", "budget:\n  max_total_cost_usd: 5.0\n  max_total_tokens: 100\n"
                                  "  max_calls: 10\n"),
                       ("TIERS", "maturity_tiers:\n  unlocked_tier: tier_1   # keep me\n")):
        p = tmp_path / f"{name.lower()}.yaml"
        p.write_text(text, encoding="utf-8")
        monkeypatch.setattr(m, name, p)
    monkeypatch.setattr(m, "APPROVER", tmp_path / "approver.txt")
    m.app.config.update(TESTING=True)
    return m


def _ok_headers(port=5001):
    return {"Host": f"127.0.0.1:{port}", "Origin": f"http://127.0.0.1:{port}",
            "Sec-Fetch-Site": "same-origin"}


# ----------------------------------------------------------------- the attack must fail
def test_a_cross_origin_post_is_refused(app):
    c = app.app.test_client()
    r = c.post("/save", data={"tier": "tier_3", "cap_cost": "9999"},
               headers={"Host": "127.0.0.1:5001", "Origin": EVIL})
    assert r.status_code == 403
    assert yaml.safe_load(app.TIERS.read_text())["maturity_tiers"]["unlocked_tier"] == "tier_1"


def test_a_post_without_a_token_is_refused_even_from_the_right_origin(app):
    c = app.app.test_client()
    r = c.post("/save", data={"tier": "tier_3"}, headers=_ok_headers())
    assert r.status_code == 403 and b"CSRF" in r.data
    assert yaml.safe_load(app.TIERS.read_text())["maturity_tiers"]["unlocked_tier"] == "tier_1"


def test_a_wrong_token_is_refused(app):
    c = app.app.test_client()
    r = c.post("/save", data={"tier": "tier_3", G.FIELD: "not-the-token"}, headers=_ok_headers())
    assert r.status_code == 403


def test_a_rebinding_host_is_refused(app):
    """An attacker domain resolving to 127.0.0.1 would pass an Origin check on its own origin, so
    the Host header has to be pinned to loopback as well."""
    c = app.app.test_client()
    r = c.post("/save", data={"tier": "tier_3", G.FIELD: G.TOKEN},
               headers={"Host": "evil.example:5001", "Origin": "http://evil.example:5001"})
    assert r.status_code == 403 and b"loopback" in r.data


def test_a_cross_site_fetch_marker_is_refused(app):
    c = app.app.test_client()
    r = c.post("/save", data={"tier": "tier_3", G.FIELD: G.TOKEN},
               headers={**_ok_headers(), "Sec-Fetch-Site": "cross-site"})
    assert r.status_code == 403


def test_the_key_deleting_endpoint_is_guarded():
    """The onboarding console can remove API keys from the OS keychain, so it is the one that
    matters most. Checked without importing it (importing loads the keychain)."""
    src = (ROOT / "scripts" / "run_onboarding.py").read_text(encoding="utf-8")
    assert "console_guard.install(app, port=PORT)" in src
    assert src.count("_CSRF") >= 7, "every form must carry the token"


# ----------------------------------------------------------------- the console must still work
def test_a_legitimate_post_succeeds_and_preserves_comments(app):
    c = app.app.test_client()
    r = c.post("/save", data={"tier": "tier_2", "cap_cost": "7.5", G.FIELD: G.TOKEN},
               headers=_ok_headers())
    assert r.status_code == 200
    tiers = app.TIERS.read_text(encoding="utf-8")
    assert yaml.safe_load(tiers)["maturity_tiers"]["unlocked_tier"] == "tier_2"
    assert "# keep me" in tiers, "the comment-preserving edit is the reason for the textual writer"
    assert yaml.safe_load(app.BUDGET.read_text())["budget"]["max_total_cost_usd"] == 7.5


def test_get_requests_are_untouched(app):
    assert app.app.test_client().get("/").status_code == 200


# ----------------------------------------------------------------- value hygiene
def test_a_newline_in_a_value_cannot_inject_a_sibling_key(tmp_path):
    """`local_model="llama3.1:8b\\n  mode: deterministic"` used to add a key to runtime.yaml."""
    p = tmp_path / "runtime.yaml"
    p.write_text("runtime:\n  local_model: llama3.1:8b\n  mode: local   # keep\n", encoding="utf-8")
    with pytest.raises(ValueError):
        G.set_scalar_checked(p, "local_model", "llama3.1:8b\n  mode: deterministic")
    doc = yaml.safe_load(p.read_text(encoding="utf-8"))
    assert doc["runtime"]["mode"] == "local"
    assert "# keep" in p.read_text(encoding="utf-8")


def test_an_edit_that_changes_a_second_key_is_rolled_back(tmp_path):
    """The verification is structural, not a blocklist: whatever the mechanism, if more than the
    target key moved, the file goes back."""
    p = tmp_path / "c.yaml"
    p.write_text("a:\n  x: 1\n  y: 2\n", encoding="utf-8")
    before = p.read_text(encoding="utf-8")
    assert G.set_scalar_checked(p, "x", 5) is True
    assert yaml.safe_load(p.read_text())["a"] == {"x": 5, "y": 2}
    with pytest.raises(ValueError):
        G.set_scalar_checked(p, "x", "3\n  y: 99")
    assert yaml.safe_load(p.read_text())["a"]["y"] == 2
    p.write_text(before, encoding="utf-8")


@pytest.mark.parametrize("bad", ["a b", "x\ny", "v'\"", "a" * 200, "", "$(whoami)", "k: v", "a,b"])
def test_unsafe_scalars_are_refused(bad):
    """The rule is about YAML and HTML injection — newlines, quotes, spaces, YAML punctuation —
    not path safety. No caller uses these values as a path (they become YAML scalars and file
    CONTENT), so `../../etc` is deliberately allowed, exactly as `/` has to be for model tags
    like `anthropic/claude-x`."""
    with pytest.raises(ValueError):
        G.safe_scalar(bad)


@pytest.mark.parametrize("good", ["tier_2", "llama3.1:8b", "gpt-4o-mini", "1.5", "Evangelos"])
def test_ordinary_values_still_pass(good):
    assert G.safe_scalar(good) == good


def test_an_allow_list_is_enforced(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text("maturity_tiers:\n  unlocked_tier: tier_1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        G.set_scalar_checked(p, "unlocked_tier", "tier_9", allowed={"tier_0", "tier_1"})
    assert yaml.safe_load(p.read_text())["maturity_tiers"]["unlocked_tier"] == "tier_1"


def test_safe_methods_are_never_refused():
    class R:
        method = "GET"
        headers: dict = {}
        form: dict = {}
    assert G.refusal(R(), 5001) is None
