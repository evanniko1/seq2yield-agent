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

from flask import Flask, render_template_string  # noqa: E402

from agents.router import runtime_mode  # noqa: E402
from orchestration import store  # noqa: E402

app = Flask(__name__)

_SHELL = """<!doctype html><meta charset=utf-8><title>seq2yield</title>
<style>
 body{font:14px/1.5 system-ui,sans-serif;margin:0;background:#0f1115;color:#d7dbe0}
 header{background:#161a21;padding:10px 18px;border-bottom:1px solid #262c36}
 header a{color:#8ab4f8;text-decoration:none;margin-right:16px;font-weight:600}
 main{padding:18px 24px;max-width:1100px}
 h1{font-size:18px;margin:.2em 0 .6em} h2{font-size:15px;color:#9aa4b2;margin-top:1.4em}
 table{border-collapse:collapse;width:100%;margin:.4em 0}
 th,td{border-bottom:1px solid #262c36;padding:6px 9px;text-align:left;vertical-align:top}
 th{color:#9aa4b2;font-weight:600} tr:hover td{background:#161a21}
 .chip{display:inline-block;padding:1px 7px;border-radius:9px;font-size:12px}
 .ok{background:#14351f;color:#79d18b} .no{background:#3a1620;color:#e88}
 .mut{color:#6b7480} a{color:#8ab4f8;text-decoration:none} code{color:#c8b6ff}
</style>
<header>
 <a href="/">Scoreboard</a><a href="/queries">Council queries</a>
 <a href="/datasets">Datasets</a><a href="/cost">Cost</a>
 <span class=mut style="float:right">provider mode: <b>{{mode}}</b></span>
</header><main>{{body|safe}}</main>"""


def _page(body: str):
    return render_template_string(_SHELL, body=body, mode=runtime_mode())


def _con():
    con = store.connect()
    store.ingest_all(con)           # idempotent refresh
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
        b.append(f"<tr><td>{t.get('dataset')}</td><td>{t.get('scope','')}</td><td><b>{t.get('winner')}</b></td>"
                 f"<td>{sig}</td><td class=mut>{t.get('selection','')}</td><td class=mut>{t.get('ts','')}</td></tr>")
    b.append("</table><h2>Accepted claims</h2><table>"
             "<tr><th>run</th><th>dataset</th><th>candidate vs baseline</th><th>ΔR²</th><th>unit</th><th>claim</th></tr>")
    for c in claims:
        b.append(f"<tr><td class=mut>{c['run_id']}</td><td>{c['dataset']}</td>"
                 f"<td>{c['candidate_model']} vs {c['baseline_model']}</td><td>{c['mean_delta_r2']}</td>"
                 f"<td class=mut>{c['bootstrap_unit']}</td><td>{c['claim']}</td></tr>")
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
        b.append(f"<tr><td><a href='/query/{q['trajectory_id']}'>{q['trajectory_id']}</a></td>"
                 f"<td class=mut>{q['ts']}</td><td>{q.get('chair_status') or ''}</td>"
                 f"<td>{q.get('chosen_proposal_id') or ''}</td><td>{q.get('n_proposals') or ''}</td></tr>")
    b.append("</table>")
    con.close()
    return _page("".join(b))


@app.route("/query/<tid>")
def query_detail(tid):
    con = _con()
    d = store.query_detail(con, tid)
    con.close()
    if not d["query"]:
        return _page(f"<h1>{tid}</h1><p class=mut>not found</p>")
    b = [f"<h1>Council query <code>{tid}</code></h1>",
         f"<p class=mut>chair: {d['query'].get('chair_status') or '—'} · "
         f"chosen: {d['query'].get('chosen_proposal_id') or '—'}</p>"]
    b.append("<h2>Proposals</h2><table><tr><th>id</th><th>model vs comparator</th><th>intervention</th>"
             "<th>dataset</th><th>hypothesis</th></tr>")
    for p in d["proposals"]:
        b.append(f"<tr><td>{p['proposal_id']}</td><td>{p['model_family']} vs {p['comparator_model']}</td>"
                 f"<td class=mut>{p['intervention_type']}</td><td>{p['dataset']}"
                 f"{'/'+p['subregion'] if p.get('subregion') not in (None,'all') else ''}</td>"
                 f"<td>{p['hypothesis']}</td></tr>")
    b.append("</table><h2>Reviews</h2><table><tr><th>proposal</th><th>role</th><th>feas</th>"
             "<th>value</th><th>clean</th><th>repro</th><th>reject</th></tr>")
    for r in d["reviews"]:
        b.append(f"<tr><td>{r['proposal_id']}</td><td>{r['role']}</td><td>{r['feasibility']}</td>"
                 f"<td>{r['scientific_value']}</td><td>{r['confoundedness']}</td>"
                 f"<td>{r['reproducibility']}</td><td class=mut>{r['reject_reason'] or ''}</td></tr>")
    b.append("</table><h2>Decision trail (RL-trace)</h2><table><tr><th>when</th><th>decision</th>"
             "<th>action</th><th>policy</th><th>reason</th></tr>")
    for e in d["events"]:
        b.append(f"<tr><td class=mut>{e['ts']}</td><td>{e['decision_type']}</td>"
                 f"<td>{e['selected_action']}</td><td class=mut>{e['policy']}</td><td>{e['reason']}</td></tr>")
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
        b.append(f"<tr><td><a href='/dataset/{d['id']}'><b>{d['id']}</b></a></td><td>{d['organism']}</td>"
                 f"<td>{d['modality']}</td><td>{d['seq_len']}</td><td>{d['structure']}</td>"
                 f"<td class=mut>{d['bootstrap_unit']}</td><td class=mut>{strata}</td><td>{ready}</td></tr>")
    b.append("</table>")
    return _page("".join(b))


@app.route("/dataset/<did>")
def dataset_card(did):
    from seq2yield.data import data_card
    c = data_card.card(did)
    b = [f"<h1>Dataset <code>{did}</code></h1>",
         f"<p class=mut>{c.get('organism')} · {c.get('modality')} · {c.get('seq_len')} bp · "
         f"{c.get('structure')} ({c.get('bootstrap_unit')} unit)</p>"]
    if c.get("note"):
        b.append(f"<p class=no chip>{c['note']}</p>")
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
            frac = " · ".join(f"{k}={v}" for k, v in bal.items())
            b.append(f"<p><b>{s}</b>: {frac}</p>")
    prov = c.get("source") or {}
    b.append(f"<h2>Provenance</h2><p class=mut>source: {prov} · citation: {c.get('citation') or '—'} "
             f"· license: {c.get('license') or '—'}</p>")
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
         f"({cpc['n_accepted_claims']} accepted; {cpc.get('by_status')})</p>"
         "<h2>By role</h2><table><tr><th>role</th><th>calls</th><th>cost</th></tr>"]
    for r in c["by_role"]:
        b.append(f"<tr><td>{r['role']}</td><td>{r['calls']}</td><td>${round(r['cost'],4)}</td></tr>")
    b.append("</table>")
    return _page("".join(b))


if __name__ == "__main__":
    print("seq2yield dashboard → http://127.0.0.1:5057  (provider mode:", runtime_mode(), ")")
    app.run(host="127.0.0.1", port=5057, debug=False)
