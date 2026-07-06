"""The provider-mode router toggle (hybrid/local/api) and the read-only progress dashboard routes.
Dashboard smoke uses the Flask test client over the real read-model (routes return 200)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from agents.router import Router  # noqa: E402


# ---- provider mode ----
def test_local_mode_forces_all_roles_to_local_tier():
    r = Router()
    r.mode = "local"
    for role in ("chair", "biology_reviewer", "modeling_reviewer"):
        assert "ollama" in r.candidates(role)              # even authority roles go local
        assert "anthropic" not in r.candidates(role)


def test_api_mode_forces_direct_providers_everywhere():
    r = Router()
    r.mode = "api"
    for role in ("modeling_reviewer", "doe_strategist"):    # normally diversity/local
        cands = r.candidates(role)
        assert set(cands) <= {"anthropic", "openai"} and cands


def test_hybrid_mode_is_the_default_split():
    r = Router()
    r.mode = "hybrid"
    assert "anthropic" in r.candidates("chair")             # authority -> direct
    assert "ollama" in r.candidates("modeling_reviewer")    # diversity -> local


# ---- dashboard routes ----
def test_progress_dashboard_routes_render():
    import run_dashboard as D
    client = D.app.test_client()
    for path in ("/", "/queries", "/datasets", "/cost"):
        resp = client.get(path)
        assert resp.status_code == 200 and len(resp.data) > 200
    assert b"Scoreboard" in client.get("/").data
    assert b"Datasets" in client.get("/datasets").data
