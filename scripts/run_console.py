"""Console hub — one entry point for the three local apps.

    python scripts/run_console.py   ->   http://127.0.0.1:5056

Boots the onboarding, operator, and dashboard apps (each on its own port) and serves a single
tabbed window so you can boot ONE thing and click your way through. The apps stay architecturally
decoupled — they coordinate only through config files + durable artifacts, never a shared process —
this hub just launches them and embeds each in a tab (an <iframe>). Killing the hub also stops the
apps it started (ones already running are left alone).

  * 🔐 Onboarding (:5058) — provider mode, keychain API keys, launch a cycle
  * 🎛 Operator   (:5001) — budgets / unlocked tier / approver, launch a cycle
  * 📊 Dashboard  (:5057) — read-only scoreboard / agent trails / cost (auto-refreshing)
"""
from __future__ import annotations

import atexit
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flask import Flask, jsonify, render_template_string  # noqa: E402

app = Flask(__name__)
HUB_PORT = 5056

APPS = [
    {"id": "onboarding", "name": "Onboarding", "emoji": "🔐", "port": 5058,
     "script": "scripts/run_onboarding.py",
     "blurb": "Pick a provider mode, store API keys in the OS keychain, set the local model, "
              "test connectivity, and launch a council cycle."},
    {"id": "operator", "name": "Operator", "emoji": "🎛", "port": 5001,
     "script": "scripts/config_app.py",
     "blurb": "Edit budgets, the unlocked tier, and selection bonuses; set the C9 approver; "
              "launch a cycle."},
    {"id": "dashboard", "name": "Dashboard", "emoji": "📊", "port": 5057,
     "script": "scripts/run_dashboard.py",
     "blurb": "Read-only scoreboard, per-query agent trails, datasets, and cost. Auto-refreshes."},
]

_children: list[subprocess.Popen] = []


def _up(port: int) -> bool:
    """Is something already listening on 127.0.0.1:port?"""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.3)
    try:
        s.connect(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def _launch_missing() -> None:
    """Start each app that isn't already listening. Ports already up are left untouched."""
    for a in APPS:
        if _up(a["port"]):
            print(f"  [up]     {a['name']:11s} :{a['port']}  (already running)")
            continue
        p = subprocess.Popen([sys.executable, str(ROOT / a["script"])], cwd=str(ROOT))
        _children.append(p)
        print(f"  [launch] {a['name']:11s} :{a['port']}  pid={p.pid}")


def _cleanup() -> None:
    for p in _children:
        try:
            p.terminate()
        except Exception:                   # noqa: BLE001
            pass


atexit.register(_cleanup)


@app.get("/status")
def status():
    return jsonify({a["id"]: _up(a["port"]) for a in APPS})


# Files a council run appends to as it works — their combined size+mtime is a cheap "something
# landed" fingerprint that lets the hub flag the Dashboard tab when you're looking elsewhere.
_SIGNAL_FILES = (
    "experiments/claims/registry.jsonl",
    "experiments/claims/tournaments.jsonl",
    "reports/model_calls.jsonl",
    "reports/decision_events.jsonl",
)


@app.get("/signal")
def signal():
    import hashlib
    h = hashlib.md5()
    for rel in _SIGNAL_FILES:
        p = ROOT / rel
        if p.exists():
            st = p.stat()
            h.update(f"{rel}:{st.st_size}:{int(st.st_mtime)}".encode())
    return jsonify({"fp": h.hexdigest()})


_SHELL = """<!doctype html><meta charset=utf-8><title>seq2yield · console</title>
<meta name=viewport content="width=device-width,initial-scale=1">
<style>
:root{
  --paper:#F5F3EE; --paper-2:#FAF9F5; --card:#FFFFFF;
  --ink:#22201C; --ink-2:#6B675F; --ink-3:#9C978C; --line:#E7E3DA; --line-2:#EFEDE6;
  --accent:#2C8C7C; --accent-ink:#1F6E62; --accent-tint:#E1F0EC;
  --ok:#3F7E4C; --no:#B0463B;
  --serif:"Iowan Old Style","Palatino Linotype",Palatino,Georgia,ui-serif,serif;
  --sans:-apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,Roboto,Helvetica,Arial,sans-serif;
  --shadow:0 1px 2px rgba(30,28,24,.05),0 8px 22px rgba(30,28,24,.05);
}
@media (prefers-color-scheme:dark){:root:not([data-theme]){
  --paper:#1A1917; --paper-2:#211F1D; --card:#252320;
  --ink:#ECE8E0; --ink-2:#A8A299; --ink-3:#787269; --line:#33302B; --line-2:#2C2A26;
  --accent:#5FC7B4; --accent-ink:#7FD6C6; --accent-tint:#1E2E2A; --ok:#6FBF7E; --no:#E08A7F;
  --shadow:0 1px 2px rgba(0,0,0,.3),0 10px 26px rgba(0,0,0,.3);
}}
:root[data-theme="dark"]{
  --paper:#1A1917; --paper-2:#211F1D; --card:#252320;
  --ink:#ECE8E0; --ink-2:#A8A299; --ink-3:#787269; --line:#33302B; --line-2:#2C2A26;
  --accent:#5FC7B4; --accent-ink:#7FD6C6; --accent-tint:#1E2E2A; --ok:#6FBF7E; --no:#E08A7F;
  --shadow:0 1px 2px rgba(0,0,0,.3),0 10px 26px rgba(0,0,0,.3);
}
*{box-sizing:border-box}
html,body{height:100%}
body{font-family:var(--sans);font-size:14.5px;margin:0;background:var(--paper);color:var(--ink);display:flex;flex-direction:column;-webkit-font-smoothing:antialiased}
header{display:flex;align-items:center;gap:6px;background:var(--paper-2);border-bottom:1px solid var(--line);padding:9px 18px;flex:none}
.brand{display:flex;align-items:center;gap:9px;margin-right:10px}
.brand-mark{width:26px;height:26px;border-radius:8px;display:grid;place-items:center;color:#fff;font-weight:700;font-size:12px;background:radial-gradient(120% 120% at 30% 20%,#57C3B1,var(--accent));box-shadow:var(--shadow)}
.brand b{font-family:var(--serif);font-size:18px;font-weight:600}
.tab{display:flex;align-items:center;gap:7px;border:1px solid transparent;background:none;color:var(--ink-2);font:inherit;font-size:13.5px;font-weight:600;padding:6px 12px;border-radius:9px;cursor:pointer;transition:.12s}
.tab:hover{background:var(--accent-tint);color:var(--accent-ink)}
.tab.active{background:var(--card);border-color:var(--line);color:var(--ink);box-shadow:var(--shadow)}
.tab .dot{width:7px;height:7px;border-radius:50%;background:var(--ink-3);flex:none}
.tab .dot.up{background:var(--ok)} .tab .dot.down{background:var(--no)}
.tab .badge{display:none;align-items:center;gap:4px;margin-left:4px;font-size:10.5px;font-weight:700;color:#fff;background:var(--accent);border-radius:20px;padding:1px 7px}
.tab .badge.show{display:inline-flex;animation:s2pulse 1.6s ease-in-out infinite}
@keyframes s2pulse{0%,100%{opacity:1}50%{opacity:.45}}
.spacer{margin-left:auto}
.theme-btn{border:1px solid var(--line);background:var(--card);color:var(--ink-2);border-radius:20px;padding:5px 12px;font:inherit;font-size:12.5px;cursor:pointer}
.theme-btn:hover{border-color:var(--accent);color:var(--accent-ink)}
.stage{flex:1;min-height:0;position:relative}
iframe{width:100%;height:100%;border:0;background:var(--paper);display:none}
iframe.show{display:block}
#home{position:absolute;inset:0;overflow:auto;padding:34px 26px;display:none}
#home.show{display:block}
.home-wrap{max-width:760px;margin:0 auto}
h1{font-family:var(--serif);font-size:24px;font-weight:600;margin:.1em 0 .1em}
.lead{color:var(--ink-2);margin:.2em 0 1.4em}
.cards{display:grid;gap:14px}
.hcard{display:flex;gap:14px;align-items:flex-start;background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px 18px;box-shadow:var(--shadow);cursor:pointer;transition:.12s}
.hcard:hover{border-color:var(--accent)}
.hcard .ico{font-size:22px;line-height:1.2;flex:none}
.hcard h3{font-family:var(--serif);font-size:16.5px;font-weight:600;margin:0 0 .2em;display:flex;align-items:center;gap:8px}
.hcard p{margin:.1em 0 0;color:var(--ink-2);font-size:13.5px;line-height:1.5}
.hcard .st{font-size:11.5px;font-weight:700;padding:1px 8px;border-radius:20px;background:var(--line-2);color:var(--ink-3)}
.hcard .st.up{background:var(--accent-tint);color:var(--accent-ink)} .hcard .st.down{background:#0000;color:var(--no)}
.note{margin-top:1.6em;color:var(--ink-3);font-size:12.5px;line-height:1.6}
</style>
<header>
 <span class=brand><span class=brand-mark>s2</span><b>seq2yield</b></span>
 <button class="tab active" data-tab=home onclick="pick('home')">Home</button>
 {{tabs|safe}}
 <span class=spacer></span>
 <button class=theme-btn id=themebtn onclick="toggleTheme()">theme</button>
</header>
<div class=stage>
 <div id=home class=show><div class=home-wrap>
  <h1>Console</h1>
  <p class=lead>One window for the three local apps. They stay decoupled — coordinating only through
   config files and durable artifacts — but you boot once and click through here.</p>
  <div class=cards>{{cards|safe}}</div>
  <div class=note>Tip: launch a council cycle from <b>Onboarding</b> or <b>Operator</b>, then open
   <b>Dashboard</b> — it auto-refreshes, so results appear as they land on disk. A grey dot means the
   app is still booting; it turns green when it is listening.</div>
 </div></div>
 {{frames|safe}}
</div>
<script>
function applyTheme(t){document.documentElement.setAttribute('data-theme',t);localStorage.setItem('s2theme',t);var b=document.getElementById('themebtn');if(b)b.textContent=(t==='dark'?'\\u2600 light':'\\u263e dark')}
function toggleTheme(){var cur=document.documentElement.getAttribute('data-theme')||(matchMedia('(prefers-color-scheme:dark)').matches?'dark':'light');applyTheme(cur==='dark'?'light':'dark')}
(function(){var s=localStorage.getItem('s2theme');if(s){applyTheme(s)}else{var d=matchMedia('(prefers-color-scheme:dark)').matches;var b=document.getElementById('themebtn');if(b)b.textContent=(d?'\\u2600 light':'\\u263e dark')}})();
function pick(id){
 document.querySelectorAll('.tab').forEach(function(t){t.classList.toggle('active',t.dataset.tab===id)});
 var home=document.getElementById('home'); home.classList.toggle('show',id==='home');
 document.querySelectorAll('iframe.appframe').forEach(function(f){
   var on=f.dataset.tab===id; f.classList.toggle('show',on);
   if(on&&!f.src){f.src=f.dataset.src;}   // lazy-load the app the first time its tab is opened
 });
 if(id==='dashboard'){var b=document.querySelector('.badge[data-badge]'); if(b)b.classList.remove('show');}
}
// Light the Dashboard tab's "new" badge when council artifacts change while you're on another tab.
var _lastFp=null;
function checkSignal(){
 fetch('/signal',{cache:'no-store'}).then(function(r){return r.json()}).then(function(s){
  if(_lastFp===null){_lastFp=s.fp;return;}          // first sample: establish a baseline, no badge
  if(s.fp!==_lastFp){
   _lastFp=s.fp;
   var active=document.querySelector('.tab.active');
   if(!active||active.dataset.tab!=='dashboard'){
    var b=document.querySelector('.badge[data-badge]'); if(b)b.classList.add('show');
   }
  }
 }).catch(function(){});
}
setInterval(checkSignal,4000); checkSignal();
function refreshStatus(){
 fetch('/status',{cache:'no-store'}).then(function(r){return r.json()}).then(function(s){
  document.querySelectorAll('.tab[data-tab]').forEach(function(t){
   var id=t.dataset.tab; if(!(id in s))return;
   var d=t.querySelector('.dot'); if(d){d.className='dot '+(s[id]?'up':'down');}
  });
  document.querySelectorAll('.hcard[data-tab]').forEach(function(c){
   var id=c.dataset.tab, st=c.querySelector('.st');
   if(st&&(id in s)){st.className='st '+(s[id]?'up':'down');st.textContent=s[id]?'running':'booting…';}
  });
 }).catch(function(){});
}
setInterval(refreshStatus,3000); refreshStatus();
</script>"""


def _render() -> str:
    tabs, cards, frames = [], [], []
    for a in APPS:
        url = f"http://127.0.0.1:{a['port']}/"
        badge = ' <span class=badge data-badge>new</span>' if a["id"] == "dashboard" else ""
        tabs.append(
            f'<button class=tab data-tab={a["id"]} onclick="pick(\'{a["id"]}\')">'
            f'<span class=dot></span>{a["emoji"]} {a["name"]}{badge}</button>')
        cards.append(
            f'<div class=hcard data-tab={a["id"]} onclick="pick(\'{a["id"]}\')">'
            f'<span class=ico>{a["emoji"]}</span><div style="flex:1">'
            f'<h3>{a["name"]} <span class=st>…</span></h3><p>{a["blurb"]}</p></div></div>')
        frames.append(
            f'<iframe class=appframe data-tab={a["id"]} data-src="{url}" '
            f'title="{a["name"]}"></iframe>')
    return render_template_string(_SHELL, tabs="".join(tabs), cards="".join(cards),
                                  frames="".join(frames))


@app.get("/")
def index():
    return _render()


if __name__ == "__main__":
    print("Starting seq2yield console apps...")
    _launch_missing()
    print(f"\nconsole hub: http://127.0.0.1:{HUB_PORT}\n")
    app.run(host="127.0.0.1", port=HUB_PORT, debug=False)
