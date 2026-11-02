"""Local progress interface (read-only) — scoreboard, per-council-query agent trails, datasets, cost.

    python scripts/ingest_history.py     # build/refresh reports/seq2yield.db
    python scripts/run_dashboard.py      # serve http://127.0.0.1:5057

Reads the SQLite read-model (orchestration/store). Ingests on startup so it is always current.
Nothing here writes experiment data — inspection only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from urllib.parse import quote  # noqa: E402

from flask import Flask, render_template_string, request  # noqa: E402
from markupsafe import escape  # noqa: E402

from agents.router import runtime_mode  # noqa: E402
from orchestration import store  # noqa: E402

app = Flask(__name__)

# EVERY value interpolated into a page is untrusted: council rows carry LLM prose (hypotheses,
# reject reasons), catalogue rows carry text harvested from publications and repository listings, and
# path parameters come from the URL. The body is rendered with `|safe`, so Jinja will not escape
# anything for us — every interpolation goes through _e(), and a DOI only becomes a link when it
# looks like a DOI (never an arbitrary string that could be a javascript: URL).
_DOI_OK = __import__("re").compile(r"^10\.\d{4,9}/[^\s\"'<>]+$")


def _e(v) -> str:
    return str(escape("" if v is None else v))


def _doi_link(doi: str | None) -> str:
    if doi and _DOI_OK.match(doi.strip()):
        return f"<a href='https://doi.org/{_e(doi.strip())}' rel=noreferrer>{_e(doi.strip())}</a>"
    return f"<span class=mut>{_e(doi) or '—'}</span>"

_SHELL = """<!doctype html><meta charset=utf-8><title>seq2yield</title>
<meta name=viewport content="width=device-width,initial-scale=1">
<style>
/* Claude-editorial aesthetic (warm paper, serif headings, soft cards) with a distinct TEAL/PINE
   accent (vs Cellarium's clay). Light + dark: system default via prefers-color-scheme, overridable
   by a header toggle that stamps data-theme on <html>. */
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
header a{color:var(--ink-2);text-decoration:none;font-weight:600;font-size:13.5px;padding:5px 10px;border-radius:8px;transition:.12s}
header a:hover{background:var(--accent-tint);color:var(--accent-ink)}
.spacer{margin-left:auto}
.mode{color:var(--ink-3);font-size:12.5px}
.theme-btn{border:1px solid var(--line);background:var(--card);color:var(--ink-2);border-radius:20px;padding:5px 12px;font:inherit;font-size:12.5px;cursor:pointer;transition:.12s}
.theme-btn:hover{border-color:var(--accent);color:var(--accent-ink)}
main{padding:24px 26px;max-width:1120px;margin:0 auto}
h1{font-family:var(--serif);font-size:23px;font-weight:600;margin:.1em 0 .5em;letter-spacing:.2px}
h2{font-family:var(--serif);font-size:16px;font-weight:600;color:var(--ink-2);margin:1.7em 0 .5em}
table{border-collapse:separate;border-spacing:0;width:100%;margin:.5em 0;background:var(--card);border:1px solid var(--line);border-radius:12px;overflow:hidden;box-shadow:var(--shadow)}
th,td{border-bottom:1px solid var(--line);padding:8px 12px;text-align:left;vertical-align:top;font-size:13.5px}
tr:last-child td{border-bottom:none}
th{color:var(--ink-3);font-weight:700;font-size:11px;text-transform:uppercase;letter-spacing:.05em;background:var(--paper-2)}
tr:hover td{background:var(--line-2)}
.chip{display:inline-block;padding:2px 9px;border-radius:20px;font-size:12px;font-weight:600}
.ok{background:var(--ok-tint);color:var(--ok)} .no{background:var(--no-tint);color:var(--no)}
.mut{color:var(--ink-3)}
a{color:var(--accent-ink);text-decoration:none} a:hover{text-decoration:underline}
code{font-family:var(--mono);font-size:.9em;color:var(--accent-ink);background:var(--accent-tint);padding:1px 5px;border-radius:5px}
.live{display:flex;align-items:center;gap:6px;color:var(--ink-3);font-size:12px;white-space:nowrap}
.live .dot{width:7px;height:7px;border-radius:50%;background:var(--ok);flex:none}
.embedded .brand,.embedded .theme-btn{display:none}   /* de-dupe chrome when embedded in the hub */
</style>
<script>if(window.self!==window.top){document.documentElement.classList.add('embedded')}</script>
<header>
 <span class=brand><span class=brand-mark>s2</span><b>seq2yield</b></span>
 <a href="/">Scoreboard</a><a href="/queries">Council queries</a><a href="/datasets">Datasets</a><a href="/catalogue">Catalogue</a><a href="/cost">Cost</a>
 <span class=spacer></span>
 <span class=mode>provider: <b>{{mode}}</b></span>
 <span class=live title="auto-refreshing every 5s"><span class=dot></span><span id=liveTime>live</span></span>
 <button class=theme-btn id=themebtn onclick="toggleTheme()">theme</button>
</header><main>{{body|safe}}</main>
<script>
function applyTheme(t){document.documentElement.setAttribute('data-theme',t);localStorage.setItem('s2theme',t);var b=document.getElementById('themebtn');if(b)b.textContent=(t==='dark'?'\\u2600 light':'\\u263e dark')}
function toggleTheme(){var cur=document.documentElement.getAttribute('data-theme')||(matchMedia('(prefers-color-scheme:dark)').matches?'dark':'light');applyTheme(cur==='dark'?'light':'dark')}
(function(){var s=localStorage.getItem('s2theme');if(s){applyTheme(s)}else{var d=matchMedia('(prefers-color-scheme:dark)').matches;var b=document.getElementById('themebtn');if(b)b.textContent=(d?'\\u2600 light':'\\u263e dark')}})();
/* Flicker-free auto-refresh: re-fetch the current page, and swap ONLY <main>'s innerHTML when the
   content actually changed. No full reload, so no white flash / scroll jump. Pauses when the tab is
   hidden; a request in flight blocks the next tick. The live label shows the last refresh time. */
(function(){
 var BUSY=false;
 function tick(){
  if(document.hidden||BUSY)return; BUSY=true;
  fetch(location.pathname+location.search,{cache:'no-store'})
   .then(function(r){return r.text()})
   .then(function(t){
     var d=new DOMParser().parseFromString(t,'text/html');
     var nx=d.querySelector('main'), cur=document.querySelector('main');
     if(nx&&cur&&nx.innerHTML!==cur.innerHTML){cur.innerHTML=nx.innerHTML;}
     var lt=document.getElementById('liveTime');
     if(lt){lt.textContent='updated '+new Date().toLocaleTimeString();}
   })
   .catch(function(){})
   .finally(function(){BUSY=false;});
 }
 setInterval(tick,5000);
})();
</script>"""


def _page(body: str):
    return render_template_string(_SHELL, body=body, mode=runtime_mode())


def _con():
    con = store.connect()
    store.ingest_all_if_changed(con)    # idempotent, and a no-op unless an artifact stream changed
    return con


@app.route("/")
def scoreboard():
    con = _con()
    claims = store.accepted_claims(con)
    tours = store.scoreboard(con)
    b = ["<h1>Scoreboard</h1><h2>Tournament winners (best model per scope)</h2><table>"
         "<tr><th>dataset</th><th>scope</th><th>winner</th><th>significant</th><th>selection</th><th>when</th></tr>"]
    for t in tours or [{"dataset": "—", "scope": "", "winner": "(no tournaments recorded yet)",
                        "winner_significant": 0, "selection": "", "ts": ""}]:
        sig = "<span class='chip ok'>yes</span>" if t.get("winner_significant") else "<span class='chip no'>no</span>"
        b.append(f"<tr><td>{_e(t.get('dataset'))}</td><td>{_e(t.get('scope',''))}</td>"
                 f"<td><b>{_e(t.get('winner'))}</b></td>"
                 f"<td>{sig}</td><td class=mut>{_e(t.get('selection',''))}</td>"
                 f"<td class=mut>{_e(t.get('ts',''))}</td></tr>")
    b.append("</table><h2>Accepted claims</h2><table>"
             "<tr><th>run</th><th>dataset</th><th>candidate vs baseline</th><th>ΔR²</th>"
             "<th>unit</th><th>vs margin</th><th>measured under</th><th>claim</th></tr>")
    for c in claims:
        # provenance column (ML-02): what the claim was compared against, how it was split, and
        # which code produced it. A claim with none of these predates 2026-11-01 — say so, rather
        # than letting it sit next to a stamped one looking equally solid.
        # ci_method joins them from 2026-11-02 (ST-2): a percentile interval is not comparable
        # with a t-interval, and the percentile one was liberal at these unit counts.
        prot = " · ".join(x for x in (_e(c["baseline_source"]), _e(c["split_strategy"]),
                                      _e(c["ci_method"]), _e(c["code_sha"])) if x) or \
            "<span class='chip no'>unstamped (pre-2026-11-01)</span>"
        if c["ci_method"] in (None, "percentile") and c["split_strategy"]:
            prot += " <span class='chip no'>percentile CI</span>"
        # ST-3: what was established ABOUT THE MARGIN. A claim with no verdict was decided by the
        # old rule, which compared the margin with a point estimate — say so rather than letting
        # it sit next to a margin-tested one looking equally established.
        mv = c["margin_verdict"]
        margin_cell = (f"{_e(mv)}" + (f" <span class=mut>(±{_e(c['margin'])})</span>"
                                      if c["margin"] is not None else "")) if mv else \
            "<span class='chip no'>point-estimate rule (pre-2026-11-02)</span>"
        b.append(f"<tr><td class=mut>{_e(c['run_id'])}</td><td>{_e(c['dataset'])}</td>"
                 f"<td>{_e(c['candidate_model'])} vs {_e(c['baseline_model'])}</td>"
                 f"<td>{_e(c['mean_delta_r2'])}</td>"
                 f"<td class=mut>{_e(c['bootstrap_unit'])}</td><td class=mut>{margin_cell}</td>"
                 f"<td class=mut>{prot}</td>"
                 f"<td>{_e(c['claim'])}</td></tr>")
    b.append("</table>")
    con.close()
    return _page("".join(b))


@app.route("/queries")
def queries():
    con = _con()
    qs = store.council_queries(con)
    b = ["<h1>Council queries</h1><table><tr><th>id</th><th>when</th><th>chair</th>"
         "<th>chosen</th><th>#proposals</th></tr>"]
    for q in qs:
        tid = _e(q["trajectory_id"])
        b.append(f"<tr><td><a href='/query/{quote(q['trajectory_id'] or '', safe='')}'>{tid}</a></td>"
                 f"<td class=mut>{_e(q['ts'])}</td><td>{_e(q.get('chair_status') or '')}</td>"
                 f"<td>{_e(q.get('chosen_proposal_id') or '')}</td>"
                 f"<td>{_e(q.get('n_proposals') or '')}</td></tr>")
    b.append("</table>")
    con.close()
    return _page("".join(b))


@app.route("/query/<tid>")
def query_detail(tid):
    con = _con()
    d = store.query_detail(con, tid)
    con.close()
    if not d["query"]:
        return _page(f"<h1>{_e(tid)}</h1><p class=mut>not found</p>")
    b = [f"<h1>Council query <code>{_e(tid)}</code></h1>",
         f"<p class=mut>chair: {_e(d['query'].get('chair_status')) or '—'} · "
         f"chosen: {_e(d['query'].get('chosen_proposal_id')) or '—'}</p>"]
    b.append("<h2>Proposals</h2><table><tr><th>id</th><th>model vs comparator</th><th>intervention</th>"
             "<th>dataset</th><th>hypothesis</th></tr>")
    for p in d["proposals"]:
        sub = f"/{_e(p['subregion'])}" if p.get("subregion") not in (None, "all") else ""
        b.append(f"<tr><td>{_e(p['proposal_id'])}</td>"
                 f"<td>{_e(p['model_family'])} vs {_e(p['comparator_model'])}</td>"
                 f"<td class=mut>{_e(p['intervention_type'])}</td><td>{_e(p['dataset'])}{sub}</td>"
                 f"<td>{_e(p['hypothesis'])}</td></tr>")
    b.append("</table><h2>Reviews</h2><table><tr><th>proposal</th><th>role</th><th>feas</th>"
             "<th>value</th><th>clean</th><th>repro</th><th>reject</th></tr>")
    for r in d["reviews"]:
        b.append(f"<tr><td>{_e(r['proposal_id'])}</td><td>{_e(r['role'])}</td>"
                 f"<td>{_e(r['feasibility'])}</td>"
                 f"<td>{_e(r['scientific_value'])}</td><td>{_e(r['confoundedness'])}</td>"
                 f"<td>{_e(r['reproducibility'])}</td>"
                 f"<td class=mut>{_e(r['reject_reason'] or '')}</td></tr>")
    b.append("</table><h2>Decision trail (RL-trace)</h2><table><tr><th>when</th><th>decision</th>"
             "<th>action</th><th>policy</th><th>reason</th></tr>")
    for e in d["events"]:
        b.append(f"<tr><td class=mut>{_e(e['ts'])}</td><td>{_e(e['decision_type'])}</td>"
                 f"<td>{_e(e['selected_action'])}</td><td class=mut>{_e(e['policy'])}</td>"
                 f"<td>{_e(e['reason'])}</td></tr>")
    b.append("</table>")
    return _page("".join(b))


@app.route("/datasets")
def datasets():
    con = _con()
    ds = store.datasets_table(con)
    con.close()
    b = ["<h1>Datasets</h1><table><tr><th>id</th><th>organism</th><th>modality</th><th>len</th>"
         "<th>structure</th><th>unit</th><th>strata</th><th>ready</th></tr>"]
    for d in ds:
        ready = "<span class='chip ok'>ready</span>" if d["ready"] else "<span class='chip no'>—</span>"
        strata = ", ".join(json.loads(d.get("strata") or "[]"))
        b.append(f"<tr><td><a href='/dataset/{quote(d['id'] or '', safe='')}'><b>{_e(d['id'])}</b></a></td>"
                 f"<td>{_e(d['organism'])}</td>"
                 f"<td>{_e(d['modality'])}</td><td>{_e(d['seq_len'])}</td><td>{_e(d['structure'])}</td>"
                 f"<td class=mut>{_e(d['bootstrap_unit'])}</td><td class=mut>{_e(strata)}</td>"
                 f"<td>{ready}</td></tr>")
    b.append("</table>")
    return _page("".join(b))


@app.route("/dataset/<did>")
def dataset_card(did):
    from seq2yield.data import data_card
    try:
        c = data_card.card(did)
    except KeyError:                                   # unknown id: say so, do not 500
        return _page(f"<h1>Dataset <code>{_e(did)}</code></h1><p class=mut>no such dataset</p>"), 404
    b = [f"<h1>Dataset <code>{_e(did)}</code></h1>",
         f"<p class=mut>{_e(c.get('organism'))} · {_e(c.get('modality'))} · {_e(c.get('seq_len'))} bp · "
         f"{_e(c.get('structure'))} ({_e(c.get('bootstrap_unit'))} unit)</p>"]
    if c.get("note"):
        b.append(f"<p class=no chip>{_e(c['note'])}</p>")
    if c.get("n"):
        t = c["target"]
        b.append(f"<h2>Distribution ({c['n']:,} rows)</h2><table>"
                 f"<tr><th>length uniform</th><td>{c['length_uniform_frac']}</td>"
                 f"<th>duplicate seqs</th><td>{c['duplicate_seq_frac']}</td></tr>"
                 f"<tr><th>target mean±std</th><td>{t['mean']} ± {t['std']}</td>"
                 f"<th>skew</th><td>{t['skew']}</td></tr>"
                 f"<tr><th>target range</th><td>[{t['min']}, {t['max']}]</td>"
                 f"<th>GC mean±std</th><td>{c['gc']['mean']} ± {c['gc']['std']}</td></tr></table>")
        b.append("<h2>Strata balance</h2>")
        for s, bal in (c.get("strata_balance") or {}).items():
            frac = " · ".join(f"{_e(k)}={_e(v)}" for k, v in bal.items())
            b.append(f"<p><b>{_e(s)}</b>: {frac}</p>")
    prov = c.get("source") or {}
    b.append(f"<h2>Provenance</h2><p class=mut>source: {_e(prov)} · "
             f"citation: {_e(c.get('citation')) or '—'} · license: {_e(c.get('license')) or '—'}</p>")
    return _page("".join(b))


@app.route("/catalogue")
def catalogue():
    """The dataset catalogue: what the corpus has, what is queued, what discovery found, and the
    organism x modality gaps. Truth lives in data/catalogue/datasets.jsonl; this is the read-model."""
    status = request.args.get("status") or None
    if status not in (None, "registered", "scheduled", "candidate", "watch", "rejected"):
        status = None
    con = _con()
    rows, cov = store.catalogue_table(con, status), store.catalogue_coverage(con)
    con.close()
    st = cov["by_status"]
    tabs = " · ".join(
        f"<a href='/catalogue{'' if s is None else '?status=' + s}'>"
        f"{'all' if s is None else s}</a> <span class=mut>"
        f"{sum(st.values()) if s is None else st.get(s, 0)}</span>"
        for s in (None, "registered", "scheduled", "candidate", "watch", "rejected"))
    b = [f"<h1>Dataset catalogue</h1><p>{tabs}</p>",
         "<p class=mut>obtainability of the datasets that are not rejected: "
         + " · ".join(f"<b>{_e(k)}</b> {v}" for k, v in sorted(cov["by_verification"].items()))
         + " — <span class=mut>VERIFIED means a per-sequence table was seen in the deposit's own "
           "file listing; DEPOSIT_ONLY means the listing was opened and held no such table.</span></p>"]

    # organism x modality coverage: in-corpus (registered/scheduled) vs discovered candidates
    corpus, cands = {}, {}
    for c in cov["cells"]:
        tgt = corpus if c["status"] in ("registered", "scheduled") else cands
        tgt[(c["organism"], c["modality"])] = tgt.get((c["organism"], c["modality"]), 0) + c["n"]
    mods = sorted({m for _, m in list(corpus) + list(cands)})
    orgs = sorted({o for o, _ in list(corpus) + list(cands)})
    b.append("<h2>Coverage and gaps (organism &times; modality)</h2>"
             "<p class=mut>bold = already in the corpus · plain = discovered candidate only</p><table><tr><th></th>"
             + "".join(f"<th>{_e(m)}</th>" for m in mods) + "</tr>")
    for o in orgs:
        cells = []
        for m in mods:
            n_c, n_d = corpus.get((o, m), 0), cands.get((o, m), 0)
            cells.append(f"<td>{'<b>' + str(n_c) + '</b>' if n_c else ''}"
                         f"{(' +' + str(n_d)) if n_d else ('' if n_c else '<span class=mut>·</span>')}</td>")
        b.append(f"<tr><th>{_e(o)}</th>{''.join(cells)}</tr>")
    b.append("</table>")

    b.append("<h2>Entries</h2><table><tr><th>id</th><th>status</th><th>fit</th><th>score</th>"
             "<th>organism</th><th>modality</th><th>assay</th><th>n</th><th>len</th>"
             "<th>deposits</th><th>obtainable</th><th>primary publication</th></tr>")
    for r in rows:
        chip = ("<span class='chip ok'>%s</span>" % _e(r["status"])) if r["status"] in ("registered", "scheduled") \
            else ("<span class='chip no'>%s</span>" % _e(r["status"])) if r["status"] == "rejected" \
            else f"<span class=chip>{_e(r['status'])}</span>"
        why = _e(r["reject_reason"] or "; ".join(json.loads(r["caveats"] or "[]")))
        dash = "<span class=mut>—</span>"
        n = f"{r['n_sequences']:,}" if r["n_sequences"] is not None else dash
        b.append(f"<tr><td><b>{_e(r['id'])}</b><br><span class=mut>{_e(r['title'])}</span>"
                 f"{('<br><span class=mut>' + why + '</span>') if why else ''}</td>"
                 f"<td>{chip}</td><td>{_e(r['fit_verdict'])}</td><td>{_e(r['score'])}</td>"
                 f"<td>{_e(r['organism'])}</td><td>{_e(r['modality'])}</td><td>{_e(r['assay'])}</td>"
                 f"<td>{n}</td><td>{_e(r['seq_len_nt']) or dash}</td>"
                 f"<td>{r['n_deposits'] or '0'}</td><td>{_e(r['verification'])}</td>"
                 f"<td>{_e(r['citation']) or dash}<br>{_doi_link(r['doi'])}</td></tr>")
    b.append("</table>")
    return _page("".join(b))


@app.route("/cost")
def cost():
    con = _con()
    c = store.cost_summary(con)
    from agents.council_metrics import cost_per_claim
    cpc = cost_per_claim(con)
    con.close()
    b = [f"<h1>Cost</h1><p>{c['n_calls']} model calls · <b>${c['total_cost_usd']}</b> · "
         f"{c['total_tokens']:,} tokens · {c['n_success']} ok</p>"
         f"<p class=mut>cost per accepted claim: <b>${cpc['cost_per_accepted_claim']}</b> "
         f"({cpc['n_accepted_claims']} accepted; {_e(cpc.get('by_status'))})</p>"
         "<h2>By role</h2><table><tr><th>role</th><th>calls</th><th>cost</th></tr>"]
    for r in c["by_role"]:
        b.append(f"<tr><td>{_e(r['role'])}</td><td>{r['calls']}</td><td>${round(r['cost'],4)}</td></tr>")
    b.append("</table>")
    return _page("".join(b))


if __name__ == "__main__":
    print("seq2yield dashboard -> http://127.0.0.1:5057  (provider mode:", runtime_mode(), ")")
    app.run(host="127.0.0.1", port=5057, debug=False)
