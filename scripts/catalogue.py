"""Dataset catalogue CLI (truth: data/catalogue/datasets.jsonl; projected into reports/seq2yield.db).

    python scripts/catalogue.py sync                      # every configs/datasets/*.yaml -> registered entry
    python scripts/catalogue.py enrich                    # resolve DOIs/PMIDs/titles + text-mined accessions
    python scripts/catalogue.py verify [--id X]           # GEO/Zenodo file listings -> VERIFIED / DEPOSIT_ONLY
    python scripts/catalogue.py check --doi 10.1038/...   # already recorded? (also --pmid/--accession/--title)
    python scripts/catalogue.py discover --plan P --out leads.json  # blind search: recall + new leads
    python scripts/catalogue.py add --file cands.jsonl    # add NEW entries, SKIP anything already recorded
    python scripts/catalogue.py list [--status candidate] # table view
    python scripts/catalogue.py stats
    python scripts/catalogue.py set-status --id X --status scheduled --reason "..."

`add` never overwrites: a candidate whose publication (DOI/PMID/title) or accession is already in the
catalogue is skipped and reported with what it matched. Lifecycle changes use `set-status`.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from seq2yield import catalogue as cat  # noqa: E402
from seq2yield.catalogue import resolve  # noqa: E402


def _print_errors(errors: list) -> None:
    if errors:
        print(f"\n{len(errors)} network error(s) — these lookups were NOT completed:")
        for e in errors[:20]:
            print("  -", e)


def cmd_sync(a) -> int:
    entries = cat.load(a.path)
    changes = cat.sync_registry(entries)
    cat.save(entries, a.path)
    print("\n".join(changes) or "registry already in sync")
    return 0


def cmd_enrich(a) -> int:
    entries, errors = cat.load(a.path), []
    for e in entries:
        if a.id and e.id != a.id:
            continue
        for i, p in enumerate(e.publications):
            if p.doi and p.pmid and p.title:
                continue
            r = resolve.resolve_doi(p.doi, role=p.role, errors=errors) if p.doi else \
                resolve.resolve_citation(p.citation, role=p.role, errors=errors)
            if r:
                e.publications[i] = r.model_copy(update={"citation": p.citation, "role": p.role})
                print(f"{e.id}: {p.citation[:50]!r} -> doi={r.doi} pmid={r.pmid} | {(r.title or '')[:60]}")
            else:
                print(f"{e.id}: {p.citation[:50]!r} -> UNRESOLVED")
        for p in e.publications:
            for acc in resolve.own_deposits(p, errors=errors):
                if acc.repository in ("GEO", "Zenodo", "ArrayExpress", "ENCODE", "EGA", "dbGaP") \
                        and cat.attach_accession(entries, e.id, acc):
                    print(f"{e.id}: + {acc.relation:9s} {acc.repository}:{acc.id}")
    cat.save(entries, a.path)
    _print_errors(errors)
    return 0


def cmd_verify(a) -> int:
    entries, errors = cat.load(a.path), []
    for e in entries:
        if a.id and e.id != a.id:
            continue
        resolve.verify_entry(e, errors=errors)
        tables = [f["name"] for acc in e.accessions if acc.relation == "deposit"
                  for f in acc.files if resolve.TABLE_EXT.search(f["name"] or "")]
        print(f"{e.id:22s} {e.verification:13s} {', '.join(tables[:3])}{' …' if len(tables) > 3 else ''}")
    cat.save(entries, a.path)
    _print_errors(errors)
    return 0


def cmd_check(a) -> int:
    probe = cat.CatalogueEntry(
        id="__probe__",
        publications=[cat.Publication(citation="probe", doi=a.doi, pmid=a.pmid, title=a.title)]
        if (a.doi or a.pmid or a.title) else [],
        accessions=[cat.Accession(repository=a.repo or "", id=a.accession)] if a.accession else [])
    hit = cat.find_match(cat.load(a.path), probe)
    print(f"RECORDED as {hit[0]} (matched on {hit[1]})" if hit else "NEW")
    return 0 if hit else 1


def cmd_add(a) -> int:
    entries = cat.load(a.path)
    lines = Path(a.file).read_text(encoding="utf-8").splitlines()
    results = []
    for ln in [x for x in lines if x.strip()]:
        cand = cat.CatalogueEntry.model_validate(json.loads(ln))
        if not cand.project_value.get("score") and cand.status != "rejected":
            cand.project_value["score"] = cat.score(cand, entries)
        res = cat.add(entries, cand)
        results.append(res)
        tag = f"skipped — already recorded as {res.id} ({res.matched_on})" if res.action == "skipped" \
            else "added"
        print(f"{cand.id:24s} {tag}")
    if not a.dry_run:
        cat.save(entries, a.path)
    c = Counter(r.action for r in results)
    print(f"\n{c.get('added', 0)} added, {c.get('skipped', 0)} skipped"
          f"{' (dry run — nothing written)' if a.dry_run else ''}")
    return 0


def cmd_discover(a) -> int:
    """Run a blind query plan (full pagination), report RECALL against the catalogue's known datasets,
    and write the UNMATCHED hits (new leads) for triage. Nothing is added here — triage decides."""
    plan = json.loads(Path(a.plan).read_text(encoding="utf-8"))["queries"]
    entries, errors, hits = cat.load(a.path), [], {}
    for name, groups in plan.items():
        rows, total = resolve.epmc_search_all(resolve.build_query(groups), max_hits=a.max_hits, errors=errors)
        for r in rows:
            key = (r.get("doi") or "").lower() or r.get("pmid") or r.get("id")
            hits.setdefault(key, {"r": r, "queries": set()})["queries"].add(name)
        print(f"{name:22s} hitCount {total:6d} fetched {len(rows):6d}{'  (CAPPED)' if total > len(rows) else ''}")
    matched, leads = {}, []
    for h in hits.values():
        r = h["r"]
        probe = cat.CatalogueEntry(id="__probe__", publications=[cat.Publication(
            citation="probe", doi=r.get("doi"), pmid=r.get("pmid"), title=r.get("title"))])
        m = cat.find_match(entries, probe)
        if m:
            matched.setdefault(m[0], set()).add(m[1])
        else:
            leads.append({"doi": r.get("doi"), "pmid": r.get("pmid"), "pmcid": r.get("pmcid"),
                          "title": r.get("title"), "authors": (r.get("authorString") or "")[:90],
                          "journal": r.get("journalTitle"), "year": r.get("pubYear"),
                          "source": r.get("source"), "pubType": r.get("pubType"),
                          "n_queries": len(h["queries"]), "queries": sorted(h["queries"])})
    known = [e for e in entries if e.status in ("registered", "scheduled", "candidate") and e.publications]
    print(f"\nRECALL against known datasets ({len(hits)} unique hits):")
    for e in known:
        print(f"  {'FOUND ' if e.id in matched else 'MISSED'} {e.id:24s} {e.status}")
    print(f"recall {sum(e.id in matched for e in known)}/{len(known)}")
    leads.sort(key=lambda x: (-x["n_queries"], -(int(x["year"]) if str(x["year"]).isdigit() else 0)))
    Path(a.out).write_text(json.dumps(leads, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{len(leads)} unmatched leads -> {a.out}")
    _print_errors(errors)
    return 0


def cmd_list(a) -> int:
    rows = [e for e in cat.load(a.path) if not a.status or e.status == a.status]
    print(f"{'id':22s} {'status':10s} {'verdict':16s} {'organism':24s} {'modality':10s} "
          f"{'assay':14s} {'n':>9s} {'len':>4s} {'verif':12s}")
    for e in rows:
        print(f"{e.id:22s} {e.status:10s} {e.fit_verdict or '-':16s} {(e.organism or '-')[:24]:24s} "
              f"{(e.modality or '-')[:10]:10s} {(e.assay or '-')[:14]:14s} "
              f"{e.n_sequences if e.n_sequences is not None else '-':>9} "
              f"{e.seq_len_nt if e.seq_len_nt is not None else '-':>4} {e.verification:12s}")
    return 0


def cmd_stats(a) -> int:
    entries = cat.load(a.path)
    print(f"{len(entries)} datasets, {sum(len(e.publications) for e in entries)} publications, "
          f"{sum(len(e.accessions) for e in entries)} accessions")
    for field in ("status", "fit_verdict", "verification", "organism", "modality", "assay"):
        print(f"  {field:13s}", dict(Counter(getattr(e, field) or "-" for e in entries).most_common()))
    return 0


def cmd_set_status(a) -> int:
    entries = cat.load(a.path)
    e = cat.set_status(entries, a.id, a.status, reason=a.reason)
    cat.save(entries, a.path)
    print(f"{e.id} -> {e.status}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--path", default=None, help="catalogue JSONL (default data/catalogue/datasets.jsonl)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sync").set_defaults(fn=cmd_sync)
    p = sub.add_parser("enrich"); p.add_argument("--id"); p.set_defaults(fn=cmd_enrich)
    p = sub.add_parser("verify"); p.add_argument("--id"); p.set_defaults(fn=cmd_verify)
    p = sub.add_parser("check")
    for k in ("--doi", "--pmid", "--title", "--accession", "--repo"):
        p.add_argument(k)
    p.set_defaults(fn=cmd_check)
    p = sub.add_parser("add"); p.add_argument("--file", required=True)
    p.add_argument("--dry-run", action="store_true"); p.set_defaults(fn=cmd_add)
    p = sub.add_parser("discover"); p.add_argument("--plan", required=True); p.add_argument("--out", required=True)
    p.add_argument("--max-hits", type=int, default=3000); p.set_defaults(fn=cmd_discover)
    p = sub.add_parser("list"); p.add_argument("--status", choices=cat.STATUSES); p.set_defaults(fn=cmd_list)
    sub.add_parser("stats").set_defaults(fn=cmd_stats)
    p = sub.add_parser("set-status"); p.add_argument("--id", required=True)
    p.add_argument("--status", required=True, choices=cat.STATUSES); p.add_argument("--reason")
    p.set_defaults(fn=cmd_set_status)
    a = ap.parse_args()
    if a.path:
        a.path = Path(a.path)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
