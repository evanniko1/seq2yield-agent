"""Onboarding console — boot the council: pick a provider mode, store API keys in the OS keychain
(BYOK), point at a local model, test connectivity, and launch a cycle.

    python scripts/run_onboarding.py     # serve http://127.0.0.1:5058

Design contract (docs/DECISIONS.md, memory: seq2yield-mvp-scope):
  * API keys go into the OS KEYCHAIN via `keyring` (Windows Credential Manager / macOS Keychain /
    Linux Secret Service) — NEVER written to `.env`, a config file, or the SQLite store. AI coding
    agents read project files; a plaintext `.env` would put every secret one `Read` from a model
    context window. See src/agents/secrets.py.
  * Provider mode + local model are written to configs/runtime.yaml as TARGETED line replacements
    so the commented source keeps its comments (same approach as the K5 config app).
  * Nothing here touches strict/protected scientific files.

Runtime key precedence (model_clients/base.py): real env var > keychain > .env.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402
from flask import Flask, render_template_string, request  # noqa: E402

from agents import secrets  # noqa: E402
from agents.model_clients import base  # noqa: E402,F401  (import triggers keychain + .env load)
from agents.schemas import ExperimentIdea  # noqa: E402

app = Flask(__name__)
RUNTIME = ROOT / "configs/runtime.yaml"
POLICY = ROOT / "configs/provider_policy.yaml"

# One tiny structured call per keyed authority provider confirms the key works + structured output
# parses (mirrors scripts/verify_keys.py). A few cents at most.
_VERIFY_SYS = "Return ONLY JSON matching the schema."
_VERIFY_USER = "Propose one tier_0 experiment to predict protein expression from 96nt DNA."

# Provider metadata (env var + cheap `reviewer` model + whether we can live-test structured output).
_MODES = ("hybrid", "local", "api")


def _providers() -> list[dict]:
    pol = (yaml.safe_load(POLICY.read_text(encoding="utf-8")) or {}).get("providers", {})
    out = []
    for name, testable in (("anthropic", True), ("openai", True), ("openrouter", False)):
        p = pol.get(name, {}) or {}
        out.append({
            "name": name,
            "env": p.get("api_key_env", ""),
            "reviewer": (p.get("models", {}) or {}).get("reviewer"),
            "testable": testable,
        })
    return out


def _runtime() -> dict:
    return (yaml.safe_load(RUNTIME.read_text(encoding="utf-8")) or {}).get("runtime", {}) if RUNTIME.exists() else {}


def _set_scalar(path: Path, key: str, value) -> bool:
    """Replace `  key: <old>` with the new value on its first match — comments preserved."""
    txt = path.read_text(encoding="utf-8")
    pat = re.compile(rf"^(\s*{re.escape(key)}:\s*)([^\n#]*)(.*)$", re.MULTILINE)
    new, n = pat.subn(
        lambda m: f"{m.group(1)}{value}{'  ' + m.group(3).strip() if m.group(3).strip() else ''}",
        txt, count=1)
    if n:
        path.write_text(new, encoding="utf-8")
    return bool(n)


def _live_test(name: str, model: str) -> tuple[bool, str]:
    """Make one cheap structured call; return (ok, detail). Never raises."""
    try:
        if name == "anthropic":
            from agents.model_clients.anthropic_client import AnthropicClient as C
        elif name == "openai":
            from agents.model_clients.openai_client import OpenAIClient as C
        else:
            return (False, "no live test for this provider")
        obj = C(model).complete_structured(
            system=_VERIFY_SYS, user=_VERIFY_USER, schema=ExperimentIdea,
            role="verify", temperature=0.0, max_tokens=200)
        return (True, f"tier={obj.maturity_tier} type={obj.intervention_type}")
    except Exception as e:                       # noqa: BLE001 — surface any provider/SDK error
        return (False, f"{type(e).__name__}: {str(e)[:160]}")


# ---------------------------------------------------------------- shell (Claude-editorial, teal) ---
_SHELL = """<!doctype html><meta charset=utf-8><title>seq2yield · onboarding</title>
<meta name=viewport content="width=device-width,initial-scale=1">
<style>
:root{
  --paper:#F5F3EE; --paper-2:#FAF9F5; --card:#FFFFFF;
  --ink:#22201C; --ink-2:#6B675F; --ink-3:#9C978C;
  --line:#E7E3DA; --line-2:#EFEDE6;
  --accent:#2C8C7C; --accent-ink:#1F6E62; --accent-tint:#E1F0EC;
  --ok:#3F7E4C; --ok-tint:#E4F0E6; --no:#B0463B; --no-tint:#F6E2DE;
  --serif:"Iowan Old Style","Palatino Linotype",Palatino,Georgia,ui-serif,serif;
  --sans:-apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,Roboto,Helvetica,Arial,sans-serif;
  --mono:"SF Mono",ui-monospace,"Cascadia Code",Consolas,monospace;
  --shadow:0 1px 2px rgba(30,28,24,.05),0 8px 22px rgba(30,28,24,.05);
}
@media (prefers-color-scheme:dark){:root:not([data-theme]){
  --paper:#1A1917; --paper-2:#211F1D; --card:#252320;
  --ink:#ECE8E0; --ink-2:#A8A299; --ink-3:#787269; --line:#33302B; --line-2:#2C2A26;
  --accent:#5FC7B4; --accent-ink:#7FD6C6; --accent-tint:#1E2E2A;
  --ok:#6FBF7E; --ok-tint:#182E1D; --no:#E08A7F; --no-tint:#331B18;
  --shadow:0 1px 2px rgba(0,0,0,.3),0 10px 26px rgba(0,0,0,.3);
}}
:root[data-theme="dark"]{
  --paper:#1A1917; --paper-2:#211F1D; --card:#252320;
  --ink:#ECE8E0; --ink-2:#A8A299; --ink-3:#787269; --line:#33302B; --line-2:#2C2A26;
  --accent:#5FC7B4; --accent-ink:#7FD6C6; --accent-tint:#1E2E2A;
  --ok:#6FBF7E; --ok-tint:#182E1D; --no:#E08A7F; --no-tint:#331B18;
  --shadow:0 1px 2px rgba(0,0,0,.3),0 10px 26px rgba(0,0,0,.3);
}
*{box-sizing:border-box}
body{font-family:var(--sans);font-size:14.5px;line-height:1.55;margin:0;background:var(--paper);color:var(--ink);-webkit-font-smoothing:antialiased}
header{display:flex;align-items:center;gap:14px;background:var(--paper-2);border-bottom:1px solid var(--line);padding:11px 22px;position:sticky;top:0;z-index:5}
.brand{display:flex;align-items:center;gap:9px;margin-right:4px}
.brand-mark{width:26px;height:26px;border-radius:8px;display:grid;place-items:center;color:#fff;font-weight:700;font-size:12px;background:radial-gradient(120% 120% at 30% 20%,#57C3B1,var(--accent));box-shadow:var(--shadow)}
.brand b{font-family:var(--serif);font-size:18px;font-weight:600;letter-spacing:.2px}
.spacer{margin-left:auto}
.mode{color:var(--ink-3);font-size:12.5px}
.theme-btn{border:1px solid var(--line);background:var(--card);color:var(--ink-2);border-radius:20px;padding:5px 12px;font:inherit;font-size:12.5px;cursor:pointer;transition:.12s}
.theme-btn:hover{border-color:var(--accent);color:var(--accent-ink)}
main{padding:24px 26px;max-width:820px;margin:0 auto}
h1{font-family:var(--serif);font-size:24px;font-weight:600;margin:.1em 0 .15em;letter-spacing:.2px}
.lead{color:var(--ink-2);margin:.1em 0 1.2em;font-size:14px}
.step{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px 20px;margin:16px 0;box-shadow:var(--shadow)}
.step h2{font-family:var(--serif);font-size:16.5px;font-weight:600;margin:0 0 .2em;display:flex;align-items:center;gap:9px}
.num{width:22px;height:22px;border-radius:50%;background:var(--accent-tint);color:var(--accent-ink);display:inline-grid;place-items:center;font-family:var(--sans);font-size:12px;font-weight:700}
.hint{color:var(--ink-3);font-size:12.5px;margin:.15em 0 .8em}
label{display:block;font-size:12.5px;color:var(--ink-2);font-weight:600;margin:.7em 0 .25em}
input[type=text],input[type=password]{width:100%;max-width:460px;padding:8px 11px;border:1px solid var(--line);border-radius:9px;background:var(--paper-2);color:var(--ink);font:inherit}
input:focus{outline:none;border-color:var(--accent);box-shadow:0 0 0 3px var(--accent-tint)}
button{padding:8px 15px;border:1px solid var(--line);background:var(--card);color:var(--ink);border-radius:9px;font:inherit;font-size:13px;font-weight:600;cursor:pointer;transition:.12s}
button:hover{border-color:var(--accent);color:var(--accent-ink)}
button.primary{background:var(--accent);border-color:var(--accent);color:#fff}
button.primary:hover{background:var(--accent-ink);color:#fff}
button.danger:hover{border-color:var(--no);color:var(--no)}
.banner{border-radius:11px;padding:11px 14px;font-size:13px;line-height:1.5;margin:.5em 0}
.embedded header{display:none}   /* hide our own header when embedded in the console hub iframe */
.banner.info{background:var(--accent-tint);color:var(--accent-ink)}
.banner.warn{background:var(--no-tint);color:var(--no)}
.banner.ok{background:var(--ok-tint);color:var(--ok)}
.radio{display:flex;gap:10px;flex-wrap:wrap;margin:.5em 0}
.opt{flex:1;min-width:180px;border:1px solid var(--line);border-radius:11px;padding:11px 13px;cursor:pointer;transition:.12s;background:var(--paper-2)}
.opt:hover{border-color:var(--accent)}
.opt input{margin-right:7px}
.opt.sel{border-color:var(--accent);background:var(--accent-tint)}
.opt b{font-size:13.5px} .opt .d{color:var(--ink-3);font-size:12px;margin-top:2px}
.keyrow{display:flex;align-items:center;justify-content:space-between;gap:10px;padding:9px 0;border-bottom:1px solid var(--line-2)}
.keyrow:last-child{border-bottom:none}
.keyname{font-weight:600;font-size:13.5px} .keyenv{color:var(--ink-3);font-size:11.5px;font-family:var(--mono)}
.chip{display:inline-block;padding:2px 9px;border-radius:20px;font-size:11.5px;font-weight:600}
.chip.ok{background:var(--ok-tint);color:var(--ok)} .chip.no{background:var(--no-tint);color:var(--no)}
.chip.env{background:var(--accent-tint);color:var(--accent-ink)}
.mono{font-family:var(--mono);font-size:12.5px}
.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:.5em 0}
.inline{display:inline} hr{border:none;border-top:1px solid var(--line);margin:1.4em 0}
a{color:var(--accent-ink)}
code{font-family:var(--mono);font-size:.88em;color:var(--accent-ink);background:var(--accent-tint);padding:1px 5px;border-radius:5px}
</style>
<script>if(window.self!==window.top){document.documentElement.classList.add('embedded')}</script>
<header>
 <span class=brand><span class=brand-mark>s2</span><b>seq2yield</b></span>
 <span style="color:var(--ink-3);font-size:13px">onboarding</span>
 <span class=spacer></span>
 <span class=mode>keychain: <b>{{backend}}</b></span>
 <button class=theme-btn id=themebtn onclick="toggleTheme()">theme</button>
</header><main>{{body|safe}}</main>
<script>
function applyTheme(t){document.documentElement.setAttribute('data-theme',t);localStorage.setItem('s2theme',t);var b=document.getElementById('themebtn');if(b)b.textContent=(t==='dark'?'\\u2600 light':'\\u263e dark')}
function toggleTheme(){var cur=document.documentElement.getAttribute('data-theme')||(matchMedia('(prefers-color-scheme:dark)').matches?'dark':'light');applyTheme(cur==='dark'?'light':'dark')}
(function(){var s=localStorage.getItem('s2theme');if(s){applyTheme(s)}else{var d=matchMedia('(prefers-color-scheme:dark)').matches;var b=document.getElementById('themebtn');if(b)b.textContent=(d?'\\u2600 light':'\\u263e dark')}})();
</script>"""


def _page(body: str):
    return render_template_string(_SHELL, body=body, backend=secrets.backend_name())


# ------------------------------------------------------------------------------- body builder ---
def _render(msg: str = "", verify_rows: list[tuple[str, bool, str]] | None = None) -> str:
    rt = _runtime()
    mode = rt.get("mode", "hybrid")
    local_model = rt.get("local_model", "llama3.1:8b")
    st = secrets.status()
    kr_ok = secrets.available()

    b = ['<h1>Boot the council</h1>',
         '<p class=lead>Pick a provider mode, store your API keys in the OS keychain, point at a '
         'local model, verify, and launch a cycle.</p>']
    if msg:
        b.append(msg)

    # keychain availability banner
    if kr_ok:
        b.append('<div class="banner info">🔐 Keys are stored in your OS keychain '
                 f'(<b>{secrets.backend_name()}</b>) — never in <code>.env</code>, a config file, or '
                 'the database. Runtime precedence: real env var → keychain → .env.</div>')
    else:
        b.append('<div class="banner warn">⚠ No OS keychain backend is available. Install it with '
                 '<code>pip install -e ".[secrets]"</code> (Windows/macOS have a built-in backend; '
                 'on Linux also install <code>libsecret</code>). Until then keys can\'t be stored '
                 'securely here — the council will fall back to any <code>.env</code> / env vars.</div>')

    # --- Step 1: provider mode -------------------------------------------------------------------
    opts = {
        "hybrid": ("Hybrid <span class=hint style=display:inline>(recommended)</span>",
                   "Diversity roles run on local Ollama; authority roles + chair on paid APIs."),
        "local": ("Local", "All roles on local Ollama. Free, no API spend, no keys needed."),
        "api": ("API", "All roles on paid providers (Anthropic / OpenAI). Highest fidelity, costs $."),
    }
    b.append('<div class=step><h2><span class=num>1</span> Provider mode</h2>'
             '<div class=hint>Written to <code>configs/runtime.yaml</code>.</div>'
             '<form method=post action=/mode><div class=radio>')
    for m in _MODES:
        title, desc = opts[m]
        sel = " sel" if m == mode else ""
        ck = " checked" if m == mode else ""
        b.append(f'<label class="opt{sel}"><input type=radio name=mode value={m}{ck}>'
                 f'<b>{title}</b><div class=d>{desc}</div></label>')
    b.append('</div><button class=primary type=submit>Set mode</button></form></div>')

    # --- Step 2: API keys (keychain) -------------------------------------------------------------
    b.append('<div class=step><h2><span class=num>2</span> API keys '
             '<span class=hint style="display:inline;font-weight:400">→ OS keychain</span></h2>')
    env_keys = secrets.dotenv_keys() if kr_ok else []
    if env_keys:
        b.append('<div class="banner warn">'
                 f'<div>📄 Found <b>{len(env_keys)}</b> API key(s) in a plaintext <code>.env</code>: '
                 f'<span class=mono>{", ".join(env_keys)}</span>. Move them into the keychain and '
                 'retire the file — keys are verified in the keychain before <code>.env</code> is '
                 'touched.</div>'
                 '<form method=post action=/migrate_env style="margin-top:.55em">'
                 '<button class=primary type=submit>Migrate .env → keychain &amp; remove file</button>'
                 '</form></div>')
    if mode == "local":
        b.append('<div class=hint>Local mode needs no API keys. Add them anyway if you plan to '
                 'switch to hybrid/api later.</div>')
    for p in _providers():
        name, env = p["name"], p["env"]
        s = st.get(env, {})
        if s.get("in_env") and not s.get("in_keychain"):
            chip = '<span class="chip env">from env / .env</span>'
        elif s.get("in_keychain"):
            chip = '<span class="chip ok">in keychain</span>'
        else:
            chip = '<span class="chip no">not set</span>'
        masked = f'<span class=mono>{s.get("masked")}</span>' if s.get("masked") else ''
        b.append(f'<div class=keyrow><div><span class=keyname>{name}</span> &nbsp;'
                 f'<span class=keyenv>{env}</span></div><div class=row>{masked} {chip}</div></div>')
        b.append(f'<form method=post action=/key class=row style="margin:.3em 0 1em">'
                 f'<input type=hidden name=env value="{env}">'
                 f'<input type=password name=value placeholder="paste {name} key" autocomplete=off '
                 f'style="max-width:340px">'
                 f'<button class=primary type=submit>Save to keychain</button>')
        if s.get("in_keychain"):
            b.append(f'</form><form method=post action=/key/delete class=inline>'
                     f'<input type=hidden name=env value="{env}">'
                     f'<button class=danger type=submit>Remove</button></form>')
        else:
            b.append('</form>')
    b.append('</div>')

    # --- Step 3: local model ---------------------------------------------------------------------
    b.append('<div class=step><h2><span class=num>3</span> Local model (Ollama)</h2>'
             '<div class=hint>Used by diversity roles in hybrid/local mode. Pull it first: '
             f'<code>ollama pull {local_model}</code>.</div>'
             '<form method=post action=/local_model class=row>'
             f'<input type=text name=local_model value="{local_model}" style="max-width:280px">'
             '<button class=primary type=submit>Set model</button></form></div>')

    # --- Step 4: verify --------------------------------------------------------------------------
    b.append('<div class=step><h2><span class=num>4</span> Test connection</h2>'
             '<div class=hint>Makes one cheap structured call per keyed authority provider '
             '(Anthropic / OpenAI) using its cheapest model — a few cents at most. Logged to '
             '<code>reports/model_calls.jsonl</code>.</div>')
    if verify_rows:
        for nm, ok, detail in verify_rows:
            chip = '<span class="chip ok">OK</span>' if ok else '<span class="chip no">FAIL</span>'
            b.append(f'<div class=keyrow><div class=keyname>{nm}</div>'
                     f'<div class=row>{chip} <span class=mono>{detail}</span></div></div>')
    b.append('<form method=post action=/verify style="margin-top:.6em">'
             '<button type=submit>Test connection (live)</button></form></div>')

    # --- Step 5: launch --------------------------------------------------------------------------
    ready = (mode == "local") or any(
        st.get(p["env"], {}).get("in_env") or st.get(p["env"], {}).get("in_keychain")
        for p in _providers() if p["testable"])
    b.append('<div class=step><h2><span class=num>5</span> Launch</h2>'
             f'<div class=hint>Runs <code>scripts/run_council.py --n 3</code> in the background '
             f'under the <b>{mode}</b> provider mode.</div>')
    if not ready:
        b.append('<div class="banner warn">No authority key is set and mode isn\'t <b>local</b> — '
                 'the council can\'t reach a paid provider. Add a key above or switch to local mode.</div>')
    b.append('<form method=post action=/launch><button class=primary type=submit>'
             'Launch council cycle</button></form></div>')

    b.append('<hr><p class=hint>After launching, watch progress in the read-only dashboard: '
             '<code>python scripts/run_dashboard.py</code> → '
             '<a href="http://127.0.0.1:5057">127.0.0.1:5057</a>. Edit budgets/tier in the operator '
             'console: <code>python scripts/config_app.py</code>.</p>')
    return "".join(b)


# --------------------------------------------------------------------------------------- routes ---
@app.get("/")
def index():
    return _page(_render())


@app.post("/mode")
def set_mode():
    m = (request.form.get("mode") or "").strip()
    if m in _MODES:
        _set_scalar(RUNTIME, "mode", m)
        return _page(_render(f'<div class="banner ok">✓ Provider mode set to <b>{m}</b>.</div>'))
    return _page(_render('<div class="banner warn">Invalid mode.</div>'))


@app.post("/key")
def save_key():
    env = (request.form.get("env") or "").strip()
    val = (request.form.get("value") or "").strip()
    if env not in secrets.MANAGED_ENV_VARS:
        return _page(_render('<div class="banner warn">Unknown key name.</div>'))
    if not val:
        return _page(_render('<div class="banner warn">No value entered.</div>'))
    try:
        secrets.set_secret(env, val)
        import os
        os.environ[env] = val                 # available immediately for this process (verify/launch)
        return _page(_render(f'<div class="banner ok">✓ Saved <b>{env}</b> to the OS keychain.</div>'))
    except Exception as e:                     # noqa: BLE001
        return _page(_render(f'<div class="banner warn">Could not store key: {e}</div>'))


@app.post("/key/delete")
def delete_key():
    env = (request.form.get("env") or "").strip()
    if env in secrets.MANAGED_ENV_VARS:
        secrets.delete(env)
        import os
        os.environ.pop(env, None)
        return _page(_render(f'<div class="banner ok">✓ Removed <b>{env}</b> from the keychain.</div>'))
    return _page(_render('<div class="banner warn">Unknown key name.</div>'))


@app.post("/migrate_env")
def migrate_env():
    rep = secrets.migrate_dotenv()
    if not rep["migrated"]:
        why = rep.get("error", "nothing to migrate")
        return _page(_render(f'<div class="banner warn">Nothing migrated: {why}.</div>'))
    parts = [f'moved <b>{", ".join(rep["migrated"])}</b> into the keychain']
    if rep["env_removed"]:
        parts.append('removed the now-empty <code>.env</code>')
    elif rep["env_updated"]:
        parts.append('stripped those keys from <code>.env</code> (other lines kept)')
    if rep["failed"]:
        parts.append(f'<b>failed</b> (left in .env): {", ".join(rep["failed"])}')
    cls = "warn" if rep["failed"] else "ok"
    return _page(_render(f'<div class="banner {cls}">✓ ' + "; ".join(parts) + ".</div>"))


@app.post("/local_model")
def set_local_model():
    lm = (request.form.get("local_model") or "").strip()
    if lm:
        _set_scalar(RUNTIME, "local_model", lm)
        return _page(_render(f'<div class="banner ok">✓ Local model set to <b>{lm}</b>.</div>'))
    return _page(_render('<div class="banner warn">No model entered.</div>'))


@app.post("/verify")
def verify():
    rows: list[tuple[str, bool, str]] = []
    import os
    for p in _providers():
        if not p["testable"]:
            continue
        if not os.environ.get(p["env"]):
            rows.append((p["name"], False, "no key set — skipped"))
            continue
        ok, detail = _live_test(p["name"], p["reviewer"])
        rows.append((p["name"], ok, detail))
    msg = '<div class="banner info">Connectivity test complete (see step 4).</div>'
    return _page(_render(msg, verify_rows=rows))


@app.post("/launch")
def launch():
    subprocess.Popen([sys.executable, str(ROOT / "scripts/run_council.py"), "--n", "3"], cwd=str(ROOT))
    return _page(_render('<div class="banner ok">✓ Launched a council cycle in the background. '
                         'Watch it in the dashboard.</div>'))


if __name__ == "__main__":
    print("onboarding console: http://127.0.0.1:5058")
    app.run(host="127.0.0.1", port=5058, debug=False)
