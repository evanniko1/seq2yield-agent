"""K5: writable-config operator app — comment-preserving edits + rendering (no server, no real
config mutation)."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import config_app as A  # noqa: E402


def test_set_scalar_preserves_comments_and_edits():
    p = Path(tempfile.mktemp(suffix=".yaml"))
    p.write_text("budget:\n  max_total_cost_usd: 10.0   # do not lose me\n  max_calls: 5000\n")
    try:
        assert A._set_scalar(p, "max_total_cost_usd", 25.0) is True
        txt = p.read_text()
        assert "max_total_cost_usd: 25.0" in txt          # value updated
        assert "# do not lose me" in txt                  # comment preserved
        assert "max_calls: 5000" in txt                   # other lines untouched
    finally:
        p.unlink()


def test_set_scalar_missing_key_is_noop():
    p = Path(tempfile.mktemp(suffix=".yaml"))
    p.write_text("a: 1\n")
    try:
        assert A._set_scalar(p, "nonexistent", 9) is False
    finally:
        p.unlink()


def test_app_renders_operator_console():
    html = A.app.test_client().get("/").get_data(as_text=True)
    assert "operator console" in html.lower()          # heading text, case-insensitive to styling
    for section in ("Selection bonuses", "Budget caps", "Unlocked tier", "datasets ready", "Recent runs"):
        assert section in html
