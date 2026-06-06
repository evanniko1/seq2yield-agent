"""M8: read-only static dashboard export (docs/PROJECT_SPEC §21).

Generates a single self-contained HTML file from the durable research trail — research
memory, the claim registry, and the question-space coverage map. It is an AUDIT UI: it reads
artifacts and owns no workflow state. Pure static HTML + inline SVG (no JS, no deps).
"""
from __future__ import annotations

import html
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from agents import question_space  # noqa: E402

_STATUS_COLOR = {"accepted": "#1a7f37", "rejected": "#b42318", "inconclusive": "#b54708",
                 "settled": "#1a7f37", "untested": "#cfd4dc"}


def _esc(x) -> str:
    return html.escape(str(x))


def _chip(text: str, color: str) -> str:
    return (f'<span style="background:{color};color:#fff;padding:1px 8px;border-radius:10px;'
            f'font-size:12px">{_esc(text)}</span>')


def _sparkline(deltas: dict, w: int = 90, h: int = 24) -> str:
    """Inline SVG sparkline of ΔR² vs train_size, with a zero reference line."""
    if not deltas:
        return ""
    items = sorted(((int(k), float(v)) for k, v in deltas.items()))
    xs = [p[0] for p in items]
    ys = [p[1] for p in items]
    lo, hi = min(ys + [0.0]), max(ys + [0.0])
    span = (hi - lo) or 1.0
    def px(i): return 3 + i * (w - 6) / max(1, len(items) - 1)
    def py(val): return h - 3 - (val - lo) / span * (h - 6)
    pts = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, (_, v) in enumerate(items))
    zero_y = py(0.0)
    color = "#1a7f37" if ys[-1] >= 0 else "#b42318"
    return (f'<svg width="{w}" height="{h}" style="vertical-align:middle">'
            f'<line x1="0" y1="{zero_y:.1f}" x2="{w}" y2="{zero_y:.1f}" stroke="#cfd4dc" stroke-dasharray="2,2"/>'
            f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.5"/></svg>')


def _coverage_matrix(cov: dict) -> str:
    """Coverage grouped by intervention_type, each with a settled/inconclusive/untested bar."""
    groups: dict[str, list] = {}
    for cid, e in cov.items():
        itype = cid.split("|", 1)[0]
        groups.setdefault(itype, []).append(e)
    blocks = []
    for itype in sorted(groups):
        es = groups[itype]
        n = len(es)
        s = sum(1 for e in es if e["status"] == "settled")
        i = sum(1 for e in es if e["status"] == "inconclusive")
        u = n - s - i
        seg = (f'<span style="display:inline-block;height:12px;width:{100*s/n:.0f}%;background:#1a7f37"></span>'
               f'<span style="display:inline-block;height:12px;width:{100*i/n:.0f}%;background:#b54708"></span>'
               f'<span style="display:inline-block;height:12px;width:{100*u/n:.0f}%;background:#cfd4dc"></span>')
        cells = " ".join(
            f'<span title="{_esc(e["describe"])}" style="display:inline-block;width:13px;height:13px;'
            f'border-radius:3px;background:{_STATUS_COLOR.get(e["status"], "#cfd4dc")}"></span>'
            for e in sorted(es, key=lambda e: e["status"]))
        blocks.append(
            f'<div style="margin-bottom:12px"><b>{_esc(itype)}</b> '
            f'<span style="color:#667085;font-size:12px">{s} settled · {i} inconclusive · {u} untested</span>'
            f'<div style="width:340px;border-radius:6px;overflow:hidden;margin:4px 0;font-size:0">{seg}</div>'
            f'<div>{cells}</div></div>')
    return "".join(blocks)


def _cost_section(cost: dict) -> str:
    if not cost:
        return ""
    rows = []
    for prov, d in sorted(cost.get("by_provider", {}).items(), key=lambda kv: -kv[1]["tokens"]):
        rows.append(f"<tr><td>{_esc(prov)}</td><td style='text-align:right'>{d['calls']}</td>"
                    f"<td style='text-align:right'>{d['tokens']:,}</td>"
                    f"<td style='text-align:right'>${d['cost_usd']:.4f}</td></tr>")
    return (f"<h2>Cost &amp; tokens ({cost['n_calls']} model calls)</h2>"
            f"<table><tr><th>provider</th><th>calls</th><th>tokens</th><th>est. cost</th></tr>"
            f"{''.join(rows)}"
            f"<tr><td><b>total</b></td><td style='text-align:right'><b>{cost['n_calls']}</b></td>"
            f"<td style='text-align:right'><b>{cost['total_tokens']:,}</b></td>"
            f"<td style='text-align:right'><b>${cost['total_cost_usd']:.4f}</b></td></tr></table>")


def build_html(records: list[dict], claims: list[dict], cost: dict | None = None) -> str:
    summ = question_space.summarize(records)
    cov = question_space.coverage(records)
    verdicts = {"accepted": 0, "rejected": 0, "inconclusive": 0}
    for r in records:
        verdicts[r.get("status", "inconclusive")] = verdicts.get(r.get("status"), 0) + 1
    accepted_claims = [c for c in claims if c.get("claim")]
    n_revisits = sum(1 for r in records if r.get("revisit"))

    # --- runs table (with scope + data-efficiency sparkline/crossover) ---
    run_rows = []
    for r in reversed(records):
        sc = _STATUS_COLOR.get(r.get("status"), "#667085")
        delta = r.get("mean_delta")
        delta_s = f"{delta:+.3f}" if isinstance(delta, (int, float)) else "—"
        cross = r.get("crossover") or {}
        de_cell = "—"
        if cross.get("deltas_by_size"):
            sup = cross.get("superior_at")
            tag = (f"superior@{sup}" if sup else
                   (f"parity@{cross.get('parity_at')}" if cross.get("parity_at") else cross.get("trend", "")))
            de_cell = f"{_sparkline(cross['deltas_by_size'])} <span style='font-size:11px;color:#667085'>{_esc(tag)}</span>"
        run_rows.append(
            f"<tr><td>{_esc(r.get('run_id', '')[:40])}</td>"
            f"<td>{_esc(r.get('intervention_type', '—'))}</td>"
            f"<td>{_esc(r.get('candidate_model'))} vs {_esc(r.get('baseline_model'))}</td>"
            f"<td>{_esc(r.get('feature_set', 'one_hot'))}/{_esc(r.get('sampling_policy', 'random'))}"
            f"/{_esc(r.get('scope', 'global'))}</td>"
            f"<td style='text-align:right'>{delta_s}{' ↻' if r.get('revisit') else ''}</td>"
            f"<td>{de_cell}</td>"
            f"<td>{_chip(r.get('status', '?'), sc)}</td>"
            f"<td>{_esc((r.get('claim_allowed') or '')[:70])}</td></tr>")

    claim_rows = [f"<li><b>{_esc(c['run_id'])}</b>: {_esc(c['claim'])}</li>"
                  for c in reversed(accepted_claims)] or ["<li><i>no accepted claims yet</i></li>"]

    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>seq2yield-agent — research dashboard</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:24px;color:#1d2939;background:#f9fafb}}
 h1{{margin:0 0 4px}} .sub{{color:#667085;margin-bottom:20px}}
 .cards{{display:flex;gap:16px;flex-wrap:wrap;margin-bottom:24px}}
 .card{{background:#fff;border:1px solid #eaecf0;border-radius:10px;padding:14px 18px;min-width:130px}}
 .card .n{{font-size:26px;font-weight:700}} .card .l{{color:#667085;font-size:13px}}
 table{{border-collapse:collapse;width:100%;background:#fff;border:1px solid #eaecf0;border-radius:10px;overflow:hidden}}
 th,td{{padding:7px 10px;border-bottom:1px solid #f0f0f0;font-size:13px;text-align:left;vertical-align:middle}}
 th{{background:#f2f4f7}} h2{{margin-top:28px}}
 .bar{{height:10px;border-radius:5px;background:#eaecf0;overflow:hidden;width:260px}} .bar>i{{display:block;height:100%;background:#1a7f37}}
 .legend span{{font-size:12px;color:#667085;margin-right:14px}}
 .sw{{display:inline-block;width:11px;height:11px;border-radius:2px;vertical-align:middle;margin-right:4px}}
</style></head><body>
<h1>seq2yield-agent — research dashboard</h1>
<div class="sub">read-only audit view · reproduces Nikolados et al. (2022) · auto-generated</div>

<div class="cards">
  <div class="card"><div class="n">{len(records)}</div><div class="l">experiments run</div></div>
  <div class="card"><div class="n" style="color:#1a7f37">{verdicts['accepted']}</div><div class="l">accepted</div></div>
  <div class="card"><div class="n" style="color:#b42318">{verdicts['rejected']}</div><div class="l">rejected</div></div>
  <div class="card"><div class="n" style="color:#b54708">{verdicts['inconclusive']}</div><div class="l">inconclusive</div></div>
  <div class="card"><div class="n">{n_revisits}</div><div class="l">revisits (more power)</div></div>
  <div class="card"><div class="n">{summ['coverage_pct']}%</div><div class="l">question-space settled
     <div class="bar"><i style="width:{summ['coverage_pct']}%"></i></div>
     {summ['settled']}/{summ['total_cells']} cells · {summ['untested']} untested</div></div>
  {f'<div class="card"><div class="n">${cost["total_cost_usd"]:.2f}</div><div class="l">est. cost · {cost["total_tokens"]:,} tokens · {cost["n_calls"]} calls</div></div>' if cost else ''}
</div>

{_cost_section(cost)}

<h2>Accepted claims</h2>
<ul>{''.join(claim_rows)}</ul>

<h2>Question-space coverage map</h2>
<div class="legend"><span><span class="sw" style="background:#1a7f37"></span>settled</span>
 <span><span class="sw" style="background:#b54708"></span>inconclusive</span>
 <span><span class="sw" style="background:#cfd4dc"></span>untested</span>
 <span>· {summ['scope_variant_cells']} scope-variant cells</span></div>
{_coverage_matrix(cov)}

<h2>Experiments ({len(records)})</h2>
<table><tr><th>run_id</th><th>intervention</th><th>candidate vs baseline</th>
 <th>feat/sampling/scope</th><th>ΔR²</th><th>data-efficiency</th><th>verdict</th><th>claim</th></tr>
 {''.join(run_rows)}</table>

<p class="sub" style="margin-top:24px">Generated by scripts/build_dashboard.py — reads
run-cards/claims/memory, owns no workflow state (PROJECT_SPEC §21). Sparklines show per-size
ΔR² vs train_size with a zero reference line.</p>
</body></html>"""


def build_dashboard(out_path: str | Path | None = None) -> Path:
    from agents import memory
    from orchestration import budget
    from seq2yield.experiments import claim_registry
    out_path = Path(out_path or ROOT / "reports" / "dashboard" / "index.html")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cost = budget.summarize(budget.load_calls())
    out_path.write_text(build_html(memory.load(), claim_registry.load(), cost=cost),
                        encoding="utf-8")
    return out_path
