"""K5 — writable-config operator app (local Flask).

Promotes the read-only dashboard to an app that EDITS the operator-facing config (selection
bonuses, budget caps, unlocked tier) and drives the loop (launch a council cycle, set the C9
conditional-approver). Config edits are TARGETED LINE REPLACEMENTS so the well-commented source
YAMLs keep their comments (ruamel not required). Read-only status: datasets, budget, open flags,
recent runs. Never edits strict/protected scientific files.

Usage: python scripts/config_app.py   ->   http://127.0.0.1:5001
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import yaml
from flask import Flask, redirect, request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agents import memory, methodology_critic  # noqa: E402
from orchestration import budget  # noqa: E402
from seq2yield.data import datasets  # noqa: E402

app = Flask(__name__)
POLICY = ROOT / "configs/council_policy.yaml"
BUDGET = ROOT / "configs/experiment_budget.yaml"
TIERS = ROOT / "configs/maturity_tiers.yaml"
APPROVER = ROOT / "configs/pending_approver.txt"      # app-owned; loop reads via --approve-conditional


def _set_scalar(path: Path, key: str, value) -> bool:
    """Replace `  key: <old>` with the new value on its first match — comments preserved."""
    txt = path.read_text(encoding="utf-8")
    pat = re.compile(rf"^(\s*{re.escape(key)}:\s*)([^\n#]*)(.*)$", re.MULTILINE)
    new, n = pat.subn(lambda m: f"{m.group(1)}{value}{'  ' + m.group(3).strip() if m.group(3).strip() else ''}", txt, count=1)
    if n:
        path.write_text(new, encoding="utf-8")
    return bool(n)


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


PAGE = """<!doctype html><meta charset=utf-8><title>seq2yield operator</title>
<style>body{{font:14px system-ui;max-width:900px;margin:2rem auto;color:#222}}
h1{{font-size:20px}} h2{{font-size:15px;margin-top:1.6rem;border-bottom:1px solid #eee}}
input{{width:90px}} .row{{margin:.3rem 0}} .chip{{background:#eef;border-radius:10px;padding:1px 8px;margin:2px;display:inline-block}}
.sev-high{{background:#fdd}} .sev-medium{{background:#fe8}} .sev-low{{background:#efe}}
button{{padding:.4rem .8rem;margin-right:.5rem}} .ok{{color:#080}} table{{border-collapse:collapse;width:100%}}
td,th{{border-bottom:1px solid #eee;padding:3px 6px;text-align:left;font-size:13px}}</style>
<h1>seq2yield — operator console <small style=color:#888>(K5)</small></h1>
{msg}
<form method=post action=/save>
<h2>Selection bonuses (chair exploration knob)</h2>{bonuses}
<h2>Budget caps</h2>
<div class=row>max cost $ <input name=cap_cost value="{cost}"></div>
<div class=row>max tokens <input name=cap_tokens value="{tokens}" style=width:130px></div>
<div class=row>max calls <input name=cap_calls value="{calls}"></div>
<h2>Unlocked tier</h2>
<div class=row><input name=tier value="{tier}" style=width:80px> (tier_0..tier_3)</div>
<p><button>Save config</button></p></form>
<form method=post action=/launch style=display:inline><button>Launch council cycle</button></form>
<form method=post action=/approve style=display:inline>
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
    bhtml = "".join(
        f"<div class=row>{k} <input name='bonus_{k}' value='{v}'></div>" for k, v in bonuses.items()) \
        or "<div class=row><i>none configured</i></div>"
    flagchips = "".join(f"<span class='chip sev-{f.get('severity')}'>{f.get('id')}</span>"
                        for f in st["flags"]) or "<i>none</i>"
    rows = "".join(f"<tr><td>{r.get('run_id','')[:34]}</td><td>{r.get('dataset','ecoli')}</td>"
                   f"<td>{r.get('status')}</td><td>{r.get('mean_delta')}</td></tr>" for r in st["recent"])
    return PAGE.format(
        msg=msg, bonuses=bhtml, cost=caps["max_total_cost_usd"], tokens=caps["max_total_tokens"],
        calls=caps["max_calls"], tier=tier, approver=st["approver"],
        ds_ready=", ".join(st["datasets_ready"]), ds_all=", ".join(st["datasets_all"]),
        spend=st["cost"]["total_cost_usd"], ncalls=st["cost"]["n_calls"],
        ntok=st["cost"]["total_tokens"], flagchips=flagchips, rows=rows)


@app.get("/")
def index():
    return _render()


@app.post("/save")
def save():
    done = []
    for k, v in request.form.items():
        if k.startswith("bonus_"):
            done.append(_set_scalar(POLICY, k[6:], float(v)))
        elif k == "cap_cost":
            done.append(_set_scalar(BUDGET, "max_total_cost_usd", float(v)))
        elif k == "cap_tokens":
            done.append(_set_scalar(BUDGET, "max_total_tokens", int(v)))
        elif k == "cap_calls":
            done.append(_set_scalar(BUDGET, "max_calls", int(v)))
        elif k == "tier" and v in ("tier_0", "tier_1", "tier_2", "tier_3"):
            done.append(_set_scalar(TIERS, "unlocked_tier", v))
    return _render(f"<p class=ok>Saved {sum(done)} config value(s). (comments preserved)</p>")


@app.post("/launch")
def launch():
    subprocess.Popen([sys.executable, str(ROOT / "scripts/run_council.py"), "--n", "3"], cwd=str(ROOT))
    return _render("<p class=ok>Launched a council cycle in the background.</p>")


@app.post("/approve")
def approve():
    name = (request.form.get("approver") or "").strip()
    if name:
        APPROVER.write_text(name, encoding="utf-8")
    return _render(f"<p class=ok>C9 approver set to '{name}'. Pass --approve-conditional {name} to the loop.</p>")


if __name__ == "__main__":
    print("operator console: http://127.0.0.1:5001")
    app.run(host="127.0.0.1", port=5001, debug=False)
