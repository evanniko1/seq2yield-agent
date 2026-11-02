"""K5 — writable-config operator app (local Flask).

Promotes the read-only dashboard to an app that EDITS the operator-facing config (selection
bonuses, budget caps, unlocked tier) and drives the loop (launch a council cycle, set the C9
conditional-approver). Config edits are TARGETED LINE REPLACEMENTS so the well-commented source
YAMLs keep their comments (ruamel not required). Read-only status: datasets, budget, open flags,
recent runs. Never edits strict/protected scientific files.

Usage: python scripts/config_app.py   ->   http://127.0.0.1:5001
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml
from flask import Flask, request
from markupsafe import escape as _e

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agents import memory, methodology_critic  # noqa: E402
from orchestration import budget, console_guard  # noqa: E402
from seq2yield.data import datasets  # noqa: E402

app = Flask(__name__)
PORT = 5001
# Every unsafe request needs this console's own CSRF token, Origin and loopback Host (audit
# AG-04): a page the user merely visits could otherwise raise the budget caps, unlock tier 3 and
# launch a paid cycle, because form posts are CORS-simple and nothing here rejected them.
console_guard.install(app, port=PORT)
POLICY = ROOT / "configs/council_policy.yaml"
BUDGET = ROOT / "configs/experiment_budget.yaml"
TIERS = ROOT / "configs/maturity_tiers.yaml"
APPROVER = ROOT / "configs/pending_approver.txt"      # app-owned; loop reads via --approve-conditional


def _set_scalar(path: Path, key: str, value, *, allowed: set | None = None) -> bool:
    """Replace `  key: <old>` on its first match — comments preserved, and the result VERIFIED.

    The textual edit is what keeps these well-commented YAMLs readable, but on its own it let a
    value carry a newline and inject a sibling key (audit AG-04). `set_scalar_checked` re-parses
    the file afterwards and restores it unless exactly this key changed."""
    return console_guard.set_scalar_checked(path, key, value, allowed=allowed)


def _load():
    pol = yaml.safe_load(POLICY.read_text(encoding="utf-8")) or {}
    caps, _, _ = budget.load_config()
    tier = (yaml.safe_load(TIERS.read_text(encoding="utf-8")) or {})["maturity_tiers"]["unlocked_tier"]
    return pol.get("selection_bonuses", {}) or {}, caps, tier


def _status():
    recs = memory.load()
    cost = budget.summarize(budget.load_calls())
    flags = methodology_critic.open_flags(recs)
    return {
        "datasets_ready": datasets.ready_ids(),
        "datasets_all": datasets.all_ids(),
        "cost": cost,
        "flags": flags[:6],
        "recent": recs[-6:][::-1],
        "approver": APPROVER.read_text().strip() if APPROVER.exists() else "(none)",
    }


_SHELL = """<!doctype html><meta charset=utf-8><title>seq2yield · operator</title>
<meta name=viewport content="width=device-width,initial-scale=1">
<style>
:root{
  --paper:#F5F3EE;--paper-2:#FAF9F5;--card:#FFFFFF;
  --ink:#22201C;--ink-2:#6B675F;--ink-3:#9C978C;--line:#E7E3DA;--line-2:#EFEDE6;
  --accent:#2C8C7C;--accent-ink:#1F6E62;--accent-tint:#E1F0EC;
  --ok:#3F7E4C;--ok-tint:#E4F0E6;--no:#B0463B;--no-tint:#F6E2DE;--warn-tint:#F6EEDD;--warn-ink:#8A6D2F;
  --serif:"Iowan Old Style","Palatino Linotype",Palatino,Georgia,ui-serif,serif;
  --sans:-apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,Roboto,Helvetica,Arial,sans-serif;
  --shadow:0 1px 2px rgba(30,28,24,.05),0 8px 22px rgba(30,28,24,.05);
}
@media (prefers-color-scheme:dark){:root:not([data-theme]){
  --paper:#1A1917;--paper-2:#211F1D;--card:#252320;
  --ink:#ECE8E0;--ink-2:#A8A299;--ink-3:#787269;--line:#33302B;--line-2:#2C2A26;
  --accent:#5FC7B4;--accent-ink:#7FD6C6;--accent-tint:#1E2E2A;
  --ok:#6FBF7E;--ok-tint:#182E1D;--no:#E08A7F;--no-tint:#331B18;--warn-tint:#2E2717;--warn-ink:#D9B25F;
  --shadow:0 1px 2px rgba(0,0,0,.3),0 10px 26px rgba(0,0,0,.3);
}}
:root[data-theme="dark"]{
  --paper:#1A1917;--paper-2:#211F1D;--card:#252320;
  --ink:#ECE8E0;--ink-2:#A8A299;--ink-3:#787269;--line:#33302B;--line-2:#2C2A26;
  --accent:#5FC7B4;--accent-ink:#7FD6C6;--accent-tint:#1E2E2A;
  --ok:#6FBF7E;--ok-tint:#182E1D;--no:#E08A7F;--no-tint:#331B18;--warn-tint:#2E2717;--warn-ink:#D9B25F;
  --shadow:0 1px 2px rgba(0,0,0,.3),0 10px 26px rgba(0,0,0,.3);
}
*{box-sizing:border-box}
body{font-family:var(--sans);font-size:14.5px;line-height:1.55;margin:0;background:var(--paper);color:var(--ink);-webkit-font-smoothing:antialiased}
header{display:flex;align-items:center;gap:12px;background:var(--paper-2);border-bottom:1px solid var(--line);padding:11px 22px;position:sticky;top:0;z-index:5}
.brand{display:flex;align-items:center;gap:9px}
.brand-mark{width:26px;height:26px;border-radius:8px;display:grid;place-items:center;color:#fff;font-weight:700;font-size:12px;background:radial-gradient(120% 120% at 30% 20%,#57C3B1,var(--accent));box-shadow:var(--shadow)}
.brand b{font-family:var(--serif);font-size:18px;font-weight:600}
.hlabel{color:var(--ink-3);font-size:13px}
.spacer{margin-left:auto}
.theme-btn{border:1px solid var(--line);background:var(--card);color:var(--ink-2);border-radius:20px;padding:5px 12px;font:inherit;font-size:12.5px;cursor:pointer}
.theme-btn:hover{border-color:var(--accent);color:var(--accent-ink)}
main{padding:22px 26px;max-width:900px;margin:0 auto}
h1{font-family:var(--serif);font-size:23px;font-weight:600;margin:.1em 0 .4em}
h1 .k5{font-family:var(--sans);font-size:12px;color:var(--ink-3);font-weight:600}
h2{font-family:var(--serif);font-size:15.5px;font-weight:600;color:var(--ink-2);margin:1.7em 0 .5em;padding-bottom:.25em;border-bottom:1px solid var(--line)}
.row{margin:.45rem 0}
input{width:90px;padding:6px 9px;border:1px solid var(--line);border-radius:8px;background:var(--paper-2);color:var(--ink);font:inherit;font-size:13px}
input:focus{outline:none;border-color:var(--accent);box-shadow:0 0 0 3px var(--accent-tint)}
button{padding:7px 13px;margin-right:.5rem;border:1px solid var(--line);background:var(--card);color:var(--ink);border-radius:9px;font:inherit;font-size:13px;font-weight:600;cursor:pointer;transition:.12s}
button:hover{border-color:var(--accent);color:var(--accent-ink)}
.ok{color:var(--ok);font-weight:600}
.no{color:var(--no);font-weight:600}
.chip{background:var(--accent-tint);color:var(--accent-ink);border-radius:20px;padding:2px 9px;margin:2px;display:inline-block;font-size:12px;font-weight:600}
.sev-high{background:var(--no-tint);color:var(--no)} .sev-medium{background:var(--warn-tint);color:var(--warn-ink)} .sev-low{background:var(--ok-tint);color:var(--ok)}
table{border-collapse:separate;border-spacing:0;width:100%;margin:.5em 0;background:var(--card);border:1px solid var(--line);border-radius:12px;overflow:hidden;box-shadow:var(--shadow)}
td,th{border-bottom:1px solid var(--line);padding:8px 12px;text-align:left;font-size:13px}
tr:last-child td{border-bottom:none}
th{color:var(--ink-3);font-weight:700;font-size:11px;text-transform:uppercase;letter-spacing:.05em;background:var(--paper-2)}
b{color:var(--ink)}
.embedded header{display:none}   /* hide our own header when embedded in the console hub iframe */
</style>
<script>if(window.self!==window.top){document.documentElement.classList.add('embedded')}</script>
<header>
 <span class=brand><span class=brand-mark>s2</span><b>seq2yield</b></span>
 <span class=hlabel>operator</span>
 <span class=spacer></span>
 <button class=theme-btn id=themebtn onclick="toggleTheme()">theme</button>
</header><main>__BODY__</main>
<script>
function applyTheme(t){document.documentElement.setAttribute('data-theme',t);localStorage.setItem('s2theme',t);var b=document.getElementById('themebtn');if(b)b.textContent=(t==='dark'?'\\u2600 light':'\\u263e dark')}
function toggleTheme(){var cur=document.documentElement.getAttribute('data-theme')||(matchMedia('(prefers-color-scheme:dark)').matches?'dark':'light');applyTheme(cur==='dark'?'light':'dark')}
(function(){var s=localStorage.getItem('s2theme');if(s){applyTheme(s)}else{var d=matchMedia('(prefers-color-scheme:dark)').matches;var b=document.getElementById('themebtn');if(b)b.textContent=(d?'\\u2600 light':'\\u263e dark')}})();
</script>"""


PAGE = """<h1>Operator console <span class=k5>K5</span></h1>
{msg}
<form method=post action=/save>{csrf}
<h2>Selection bonuses (chair exploration knob)</h2>{bonuses}
<h2>Budget caps</h2>
<div class=row>max cost $ <input name=cap_cost value="{cost}"></div>
<div class=row>max tokens <input name=cap_tokens value="{tokens}" style=width:130px></div>
<div class=row>max calls <input name=cap_calls value="{calls}"></div>
<h2>Unlocked tier</h2>
<div class=row><input name=tier value="{tier}" style=width:80px> (tier_0..tier_3)</div>
<p><button>Save config</button></p></form>
<form method=post action=/launch style=display:inline>{csrf}<button>Launch council cycle</button></form>
<form method=post action=/approve style=display:inline>{csrf}
<input name=approver placeholder="approver name" style=width:140px><button>Set C9 approver</button></form>
<span style=margin-left:1rem>current approver: <b>{approver}</b></span>
<h2>Status</h2>
<div class=row>datasets ready: {ds_ready} &nbsp;|&nbsp; registered: {ds_all}</div>
<div class=row>spend: <b>${spend:.4f}</b> over {ncalls} calls ({ntok:,} tokens)</div>
<div class=row>open methodology flags: {flagchips}</div>
<h2>Recent runs</h2><table><tr><th>run</th><th>dataset</th><th>status</th><th>ΔR²</th></tr>{rows}</table>
"""


def _render(msg=""):
    bonuses, caps, tier = _load()
    st = _status()
    # Everything interpolated here is escaped: the approver name is operator-supplied, and the
    # run rows and methodology flags carry LLM-written text (audit IM-02/AG-04).
    bhtml = "".join(
        f"<div class=row>{_e(k)} <input name='bonus_{_e(k)}' value='{_e(v)}'></div>"
        for k, v in bonuses.items()) or "<div class=row><i>none configured</i></div>"
    flagchips = "".join(f"<span class='chip sev-{_e(f.get('severity'))}'>"
                        f"{_e(f.get('id'))}</span>"
                        for f in st["flags"]) or "<i>none</i>"
    rows = "".join(f"<tr><td>{_e(r.get('run_id', '')[:34])}</td>"
                   f"<td>{_e(r.get('dataset', 'ecoli'))}</td>"
                   f"<td>{_e(r.get('status'))}</td><td>{_e(r.get('mean_delta'))}</td></tr>"
                   for r in st["recent"])
    body = PAGE.format(
        csrf=console_guard.hidden_field(), msg=msg, bonuses=bhtml, cost=caps["max_total_cost_usd"], tokens=caps["max_total_tokens"],
        calls=caps["max_calls"], tier=_e(tier), approver=_e(st["approver"]),
        ds_ready=", ".join(st["datasets_ready"]), ds_all=", ".join(st["datasets_all"]),
        spend=st["cost"]["total_cost_usd"], ncalls=st["cost"]["n_calls"],
        ntok=st["cost"]["total_tokens"], flagchips=flagchips, rows=rows)
    return _SHELL.replace("__BODY__", body)


@app.get("/")
def index():
    return _render()


@app.post("/save")
def save():
    done = []
    try:
        for k, v in request.form.items():
            if k.startswith("bonus_"):
                done.append(_set_scalar(POLICY, console_guard.safe_scalar(k[6:]), float(v)))
            elif k == "cap_cost":
                done.append(_set_scalar(BUDGET, "max_total_cost_usd", float(v)))
            elif k == "cap_tokens":
                done.append(_set_scalar(BUDGET, "max_total_tokens", int(v)))
            elif k == "cap_calls":
                done.append(_set_scalar(BUDGET, "max_calls", int(v)))
            elif k == "tier":
                done.append(_set_scalar(TIERS, "unlocked_tier", v,
                                        allowed={"tier_0", "tier_1", "tier_2", "tier_3"}))
    except ValueError as e:      # a refused value leaves every config as it was
        return _render(f"<p class=no>Refused: {_e(str(e))}</p>")
    return _render(f"<p class=ok>Saved {sum(done)} config value(s). (comments preserved)</p>")


@app.post("/launch")
def launch():
    subprocess.Popen([sys.executable, str(ROOT / "scripts/run_council.py"), "--n", "3"], cwd=str(ROOT))
    return _render("<p class=ok>Launched a council cycle in the background.</p>")


@app.post("/approve")
def approve():
    raw = (request.form.get("approver") or "").strip()
    if not raw:
        return _render("<p class=no>Refused: empty approver name.</p>")
    try:              # this name is read back by the loop as an identity AND rendered here
        name = console_guard.safe_scalar(raw)
    except ValueError as e:
        return _render(f"<p class=no>Refused: {_e(str(e))}</p>")
    APPROVER.write_text(name, encoding="utf-8")
    return _render(f"<p class=ok>C9 approver set to '{_e(name)}'. Pass --approve-conditional "
                   f"{_e(name)} to the loop.</p>")


if __name__ == "__main__":
    print("operator console: http://127.0.0.1:5001")
    app.run(host="127.0.0.1", port=5001, debug=False)
