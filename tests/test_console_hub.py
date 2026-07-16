"""Console hub — the single entry point that launches + embeds the three local apps.

Data-free and CI-safe: rendering is static HTML and /status just probes localhost ports (all down on
CI -> all False). We never call _launch_missing (that spawns subprocesses; it only runs under
__main__)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_hub():
    spec = importlib.util.spec_from_file_location("run_console", ROOT / "scripts/run_console.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_hub_page_has_a_tab_and_iframe_per_app():
    m = _load_hub()
    html = m.app.test_client().get("/").data.decode("utf-8", "replace")
    assert m.app.test_client().get("/").status_code == 200
    for a in m.APPS:
        assert a["name"] in html
        assert f'data-src="http://127.0.0.1:{a["port"]}/"' in html      # lazy-loaded app iframe
    assert "function pick" in html and "refreshStatus" in html          # tab-switch + status poll


def test_status_endpoint_reports_every_app():
    m = _load_hub()
    js = m.app.test_client().get("/status").get_json()
    assert set(js.keys()) == {a["id"] for a in m.APPS}
    assert all(isinstance(v, bool) for v in js.values())


def test_signal_endpoint_returns_a_stable_fingerprint():
    m = _load_hub()
    c = m.app.test_client()
    fp1 = c.get("/signal").get_json()["fp"]
    assert isinstance(fp1, str) and len(fp1) == 32                    # md5 hex
    assert c.get("/signal").get_json()["fp"] == fp1                  # stable when nothing changed


def test_only_the_dashboard_tab_carries_the_new_badge():
    m = _load_hub()
    html = m.app.test_client().get("/").data.decode("utf-8", "replace")
    assert html.count("badge data-badge") == 1                       # exactly one badge...
    # ...and it sits on the dashboard tab button
    dash_btn = html.split('data-tab=dashboard')[1].split("</button>")[0]
    assert "badge data-badge" in dash_btn
