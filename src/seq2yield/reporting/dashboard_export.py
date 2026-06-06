"""M8: read-only static dashboard export (docs/PROJECT_SPEC §21).

Generates a single self-contained HTML file from the durable research trail — research
memory, the claim registry, the question-space coverage map, and the baseline registry. It is
an AUDIT UI: it reads artifacts and owns no workflow state.
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from agents import question_space  # noqa: E402

_STATUS_COLOR = {"accepted": "#1a7f37", "rejected": "#b42318", "inconclusive": "#b54708",
                 "settled": "#1a7f37", "untested": "#667085"}


def _esc(x) -> str:
    return html.escape(str(x))


def _chip(text: str, color: str) -> str:
    return (f'<span style="background:{color};color:#fff;padding:1px 8px;border-radius:10px;'
            f'font-size:12px">{_esc(text)}</span>')


def build_html(records: list[dict], claims: list[dict]) -> str:
    summ = question_space.summarize(records)
    cov = question_space.coverage(records)
    verdicts = {"accepted": 0, "rejected": 0, "inconclusive": 0}
    for r in records:
        verdicts[r.get("status", "inconclusive")] = verdicts.get(r.get("status"), 0) + 1
    accepted_claims = [c for c in claims if c.get("claim")]

    # --- runs table ---
    run_rows = []
    for r in reversed(records):
        sc = _STATUS_COLOR.get(r.get("status"), "#667085")
        delta = r.get("mean_delta")
        delta_s = f"{delta:+.3f}" if isinstance(delta, (int, float)) else "—"
        run_rows.append(
            f"<tr><td>{_esc(r.get('run_id', '')[:42])}</td>"
            f"<td>{_esc(r.get('intervention_type', '—'))}</td>"
            f"<td>{_esc(r.get('candidate_model'))} vs {_esc(r.get('baseline_model'))}</td>"
            f"<td>{_esc(r.get('feature_set', 'one_hot'))}/{_esc(r.get('sampling_policy', 'random'))}</td>"
            f"<td style='text-align:right'>{delta_s}</td>"
            f"<td>{_chip(r.get('status', '?'), sc)}</td>"
            f"<td>{_esc((r.get('claim_allowed') or '')[:80])}</td></tr>")

    # --- coverage table ---
    cov_rows = []
    for cid, e in sorted(cov.items(), key=lambda kv: (kv[1]["status"], kv[0])):
        sc = _STATUS_COLOR.get(e["status"], "#667085")
        cov_rows.append(
            f"<tr><td>{_esc(cid)}</td><td>{_esc(e['describe'])}</td>"
            f"<td>{_chip(e['status'], sc)}</td>"
            f"<td style='text-align:right'>{e['n_runs']}</td></tr>")

    # --- claims ---
    claim_rows = [f"<li><b>{_esc(c['run_id'])}</b>: {_esc(c['claim'])}</li>"
                  for c in reversed(accepted_claims)] or ["<li><i>no accepted claims yet</i></li>"]

    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>seq2yield-agent — research dashboard</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:24px;color:#1d2939;background:#f9fafb}}
 h1{{margin:0 0 4px}} .sub{{color:#667085;margin-bottom:20px}}
 .cards{{display:flex;gap:16px;flex-wrap:wrap;margin-bottom:24px}}
 .card{{background:#fff;border:1px solid #eaecf0;border-radius:10px;padding:14px 18px;min-width:140px}}
 .card .n{{font-size:26px;font-weight:700}} .card .l{{color:#667085;font-size:13px}}
 table{{border-collapse:collapse;width:100%;background:#fff;border:1px solid #eaecf0;border-radius:10px;overflow:hidden}}
 th,td{{padding:7px 10px;border-bottom:1px solid #f0f0f0;font-size:13px;text-align:left}}
 th{{background:#f2f4f7}} h2{{margin-top:28px}}
 .bar{{height:10px;border-radius:5px;background:#eaecf0;overflow:hidden;width:260px}}
 .bar>i{{display:block;height:100%;background:#1a7f37}}
</style></head><body>
<h1>seq2yield-agent — research dashboard</h1>
<div class="sub">read-only audit view · reproduces Nikolados et al. (2022) · auto-generated</div>

<div class="cards">
  <div class="card"><div class="n">{len(records)}</div><div class="l">experiments run</div></div>
  <div class="card"><div class="n" style="color:#1a7f37">{verdicts['accepted']}</div><div class="l">accepted</div></div>
  <div class="card"><div class="n" style="color:#b42318">{verdicts['rejected']}</div><div class="l">rejected</div></div>
  <div class="card"><div class="n" style="color:#b54708">{verdicts['inconclusive']}</div><div class="l">inconclusive</div></div>
  <div class="card"><div class="n">{summ['coverage_pct']}%</div><div class="l">question-space settled
     <div class="bar"><i style="width:{summ['coverage_pct']}%"></i></div>
     {summ['settled']}/{summ['total_cells']} cells · {summ['untested']} untested</div></div>
</div>

<h2>Accepted claims</h2>
<ul>{''.join(claim_rows)}</ul>

<h2>Experiments ({len(records)})</h2>
<table><tr><th>run_id</th><th>intervention</th><th>candidate vs baseline</th>
 <th>feat/sampling</th><th>ΔR²</th><th>verdict</th><th>claim</th></tr>
 {''.join(run_rows)}</table>

<h2>Question-space coverage ({summ['settled']}/{summ['total_cells']} settled,
 {summ['inconclusive']} inconclusive, {summ['untested']} untested)</h2>
<table><tr><th>cell</th><th>question</th><th>status</th><th>runs</th></tr>
 {''.join(cov_rows)}</table>

<p class="sub" style="margin-top:24px">Generated by scripts/build_dashboard.py — the dashboard
reads run-cards/claims/memory and owns no workflow state (PROJECT_SPEC §21).</p>
</body></html>"""


def build_dashboard(out_path: str | Path | None = None) -> Path:
    from agents import memory
    from seq2yield.experiments import claim_registry
    out_path = Path(out_path or ROOT / "reports" / "dashboard" / "index.html")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_html(memory.load(), claim_registry.load()), encoding="utf-8")
    return out_path
