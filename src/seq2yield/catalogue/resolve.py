"""Network resolvers for the catalogue — Europe PMC lookups, text-mined data accessions, and
deposit verification (GEO file lists incl. RAW.tar contents via filelist.txt; Zenodo file lists).

Read-only public APIs, stdlib only. Every function returns None/[] on a network failure AND reports
it through the `errors` list it is given (never silently), so a scan can say what it did not cover.
Fetched text is untrusted data: we extract identifiers with regexes; nothing is executed or obeyed.
"""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request

from .model import Accession, Publication

UA = "seq2yield-catalogue/1.0 (dataset catalogue curation; read-only)"
EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
TABLE_EXT = re.compile(r"\.(csv|tsv|txt|tab|xlsx?)(\.gz|\.zip)?$", re.I)

# data-repository accession patterns (text-mining returns rsIDs, DOIs etc. — keep only deposits)
_ACC_PATTERNS = [
    ("GEO", re.compile(r"^GSE\d+$")),
    ("ArrayExpress", re.compile(r"^E-[A-Z]{4}-\d+$")),
    ("BioProject", re.compile(r"^PRJ(NA|EB|DB)\d+$")),
    ("SRA", re.compile(r"^(SRP|ERP|DRP)\d+$")),
    ("ENCODE", re.compile(r"^ENCSR[0-9A-Z]+$")),
    ("Zenodo", re.compile(r"^10\.5281/zenodo\.\d+$", re.I)),
    ("EGA", re.compile(r"^EGA[SD]\d+$")),
    ("dbGaP", re.compile(r"^phs\d+(\.v\d+)?(\.p\d+)?$")),
]


def _get(url: str, *, as_json: bool = True, errors: list | None = None, retries: int = 3):
    if "ncbi.nlm.nih.gov" in url:
        time.sleep(0.4)                                # NCBI: <= 3 requests/s without an API key
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=30) as r:
                body = r.read().decode("utf-8", "replace")
            return json.loads(body) if as_json else body
        except urllib.error.HTTPError as e:
            # NCBI's file host answers 403 (not 429) when it is throttling a client, so back off on it
            # too — but only there, and only a few times; a persistent 403 is reported, never guessed at.
            throttled = e.code in (429, 500, 502, 503) or (e.code == 403 and "ftp.ncbi.nlm.nih.gov" in url)
            if throttled and attempt < retries:
                time.sleep(2 ** attempt * 1.5)         # back off on rate limits / transient errors
                continue
            err = e
        except (urllib.error.URLError, TimeoutError, ValueError) as e:
            err = e
        if errors is not None:
            errors.append(f"{url[:90]} -> {type(err).__name__}: {str(err)[:120]}")
        return None


def epmc_search(query: str, *, page_size: int = 25, result_type: str = "lite",
                errors: list | None = None) -> list[dict]:
    d = _get(f"{EPMC}?query={urllib.parse.quote(query)}&format=json&pageSize={page_size}"
             f"&resultType={result_type}", errors=errors)
    return (d or {}).get("resultList", {}).get("result", [])


def epmc_search_all(query: str, *, max_hits: int = 5000, errors: list | None = None) -> tuple[list[dict], int]:
    """Every hit for `query` (cursor pagination, 1000/page) up to max_hits — no relevance cutoff.
    (A fixed top-N silently drops relevant papers that rank low on broad queries.) Returns (hits, hitCount)."""
    out, cursor, total = [], "*", 0
    while len(out) < max_hits:
        d = _get(f"{EPMC}?query={urllib.parse.quote(query)}&format=json&pageSize=1000&resultType=lite"
                 f"&cursorMark={urllib.parse.quote(cursor)}", errors=errors)
        if not d:
            break
        total = d.get("hitCount", total)
        page = d.get("resultList", {}).get("result", [])
        out.extend(page)
        nxt = d.get("nextCursorMark")
        if not page or not nxt or nxt == cursor:
            break
        cursor = nxt
    return out[:max_hits], total


def build_query(groups: list[list[str]]) -> str:
    """Declarative plan -> Europe PMC syntax over TITLE+ABSTRACT. Groups are AND-ed, terms OR-ed.
    Multi-word terms are exact phrases; single words may end in '*' (prefix wildcard, e.g. random*)."""
    def term(t: str) -> str:
        if " " not in t and t.endswith("*"):
            return f"(TITLE:{t} OR ABSTRACT:{t})"
        return f'(TITLE:"{t}" OR ABSTRACT:"{t}")'
    return " AND ".join("(" + " OR ".join(term(t) for t in g) + ")" for g in groups)


def _pub(r: dict, citation: str | None = None, role: str = "primary") -> Publication:
    first = (r.get("authorString") or "").split(",")[0].strip()
    surname = first.split(" ")[0] if first else "Unknown"
    year = int(r["pubYear"]) if str(r.get("pubYear", "")).isdigit() else None
    journal = r.get("journalTitle") or ("bioRxiv/preprint" if r.get("source") == "PPR" else "")
    return Publication(citation=citation or f"{surname} et al., {journal} ({year})",
                       doi=r.get("doi"), pmid=r.get("pmid"), pmcid=r.get("pmcid"),
                       title=(r.get("title") or "").rstrip("."), year=year, role=role)


def resolve_doi(doi: str, *, role: str = "primary", errors: list | None = None) -> Publication | None:
    hits = epmc_search(f'DOI:"{doi}"', page_size=3, errors=errors)
    return _pub(hits[0], role=role) if hits else None


_NOT_ARTICLE = re.compile(r"^(author correction|correction|erratum|retraction|corrigendum)", re.I)


def parse_citation(citation: str) -> dict:
    """'de Boer et al., Nat Biotechnol 38:56 (2020)' ->
    {surname:'de Boer', journal:'Nat Biotechnol', volume:'38', page:'56', year:'2020'}."""
    head = citation.strip().split(",")[0]
    surname = re.sub(r"\s+et\s+al\.?$", "", head.strip()).strip()
    year = re.search(r"\((\d{4})\)", citation)
    m = re.search(r"et al\.?,\s*([A-Za-z][A-Za-z .&]*?)\s+(\d+):(\d+)", citation)
    journal = m.group(1).strip() if m else None
    if not m:
        mj = re.search(r"et al\.?,\s*([A-Za-z][A-Za-z .&]*?)\s*\(\d{4}\)", citation)
        journal = mj.group(1).strip() if mj else None
    return {"surname": surname, "journal": journal, "volume": m.group(2) if m else None,
            "page": m.group(3) if m else None, "year": year.group(1) if year else None}


def resolve_citation(citation: str, *, role: str = "primary",
                     errors: list | None = None) -> Publication | None:
    """Resolve a citation string to a Publication — but ONLY on positive evidence. A wrong DOI is worse
    than none (it would make the dedupe skip genuinely new datasets), so: the FIRST author must match,
    and if the citation carries volume:page those must match too; otherwise exactly one non-correction
    first-author hit in the right journal+year is required. Ambiguous -> None (UNRESOLVED)."""
    m_doi = re.search(r"10\.\d{4,9}/[^\s;,)]+", citation)
    if m_doi:
        p = resolve_doi(m_doi.group(0), role=role, errors=errors)
        if p:
            return p.model_copy(update={"citation": citation})
    c = parse_citation(citation)
    if not c["surname"] or not c["year"]:
        return None
    base = f'AUTH:"{c["surname"]}" AND PUB_YEAR:{c["year"]}'
    hits = epmc_search(f'{base} AND JOURNAL:"{c["journal"]}"', page_size=50, errors=errors) \
        if c["journal"] else []
    if not hits:
        hits = epmc_search(base, page_size=100, errors=errors)
    sn = c["surname"].lower()
    firsts = [r for r in hits if (r.get("authorString") or "").lower().startswith(sn)
              and not _NOT_ARTICLE.match(r.get("title") or "")]
    if c["volume"] and c["page"]:
        exact = [r for r in firsts if str(r.get("journalVolume") or "") == c["volume"]
                 and (r.get("pageInfo") or "").split("-")[0].strip() == c["page"]]
        return _pub(exact[0], citation=citation, role=role) if len(exact) == 1 else None
    return _pub(firsts[0], citation=citation, role=role) if len(firsts) == 1 else None


def text_mined_accessions(pmcid: str | None, *, errors: list | None = None) -> list[Accession]:
    """Data-deposit accessions text-mined from an open-access full text (Europe PMC annotations)."""
    if not pmcid:
        return []
    d = _get("https://www.ebi.ac.uk/europepmc/annotations_api/annotationsByArticleIds"
             f"?articleIds=PMC:{pmcid.upper().replace('PMC', '')}&type=Accession%20Numbers&format=JSON",
             errors=errors)
    seen, out = set(), []
    for art in d or []:
        for a in art.get("annotations", []):
            exact = (a.get("exact") or "").strip()
            for repo, pat in _ACC_PATTERNS:
                if pat.match(exact) and exact.upper() not in seen:
                    seen.add(exact.upper())
                    acc_id = exact.split("zenodo.")[-1] if repo == "Zenodo" else exact
                    out.append(Accession(repository=repo, id=acc_id, relation="mentioned"))
    return out


def geo_pubmed_ids(gse: str, *, errors: list | None = None) -> list[str]:
    """PMIDs GEO itself links to a series — the ownership test for a text-mined accession."""
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    s = _get(f"{base}/esearch.fcgi?db=gds&term={gse.upper()}[ACCN]+AND+gse[ETYP]&retmode=json",
             errors=errors)
    ids = (s or {}).get("esearchresult", {}).get("idlist", [])
    if not ids:
        return []
    d = _get(f"{base}/esummary.fcgi?db=gds&id={ids[0]}&retmode=json", errors=errors)
    return [str(x) for x in ((d or {}).get("result", {}).get(ids[0], {}).get("pubmedids") or [])]


def own_deposits(pub: Publication, *, errors: list | None = None) -> list[Accession]:
    """Text-mined accessions of `pub`, with GEO series promoted to `deposit` only when GEO links the
    series back to this paper's PMID. Everything else stays `mentioned` (a lead, not a dedupe key)."""
    out = []
    for acc in text_mined_accessions(pub.pmcid, errors=errors):
        if acc.repository == "GEO" and pub.pmid and pub.pmid in geo_pubmed_ids(acc.id, errors=errors):
            acc = acc.model_copy(update={"relation": "deposit"})
        out.append(acc)
    return out


def verify_geo(gse: str, *, errors: list | None = None) -> list[dict] | None:
    """File list for a GEO series: series-level supplementary files + RAW.tar contents (filelist.txt).

    `None` means the listing could NOT be read (throttled/offline); `[]` means it was read and the
    series has no supplementary files. The two are different facts and never share a label."""
    gse = gse.upper()
    stub = (gse[:-3] + "nnn") if len(gse) > 6 else "GSEnnn"
    base = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{stub}/{gse}/suppl/"
    html = _get(base, as_json=False, errors=errors)
    if not html:
        return None
    files = [{"name": n, "size": s} for n, _, s in
             re.findall(r'<a href="([^"/?][^"]*)">[^<]*</a>\s+(\S+ \S+)\s+(\S+)', html)
             if n != "filelist.txt"]
    if 'href="filelist.txt"' in html:
        time.sleep(0.34)
        fl = _get(base + "filelist.txt", as_json=False, errors=errors) or ""
        for ln in fl.splitlines():
            parts = ln.split("\t")
            if len(parts) >= 5 and parts[0] == "File":
                files.append({"name": parts[1], "size": parts[3], "in": "RAW.tar"})
    return files


def verify_zenodo(record: str, *, errors: list | None = None) -> list[dict] | None:
    """Zenodo record file list, or None when the record could not be read (see verify_geo)."""
    rid = re.sub(r"\D", "", record.split("zenodo.")[-1])
    d = _get(f"https://zenodo.org/api/records/{rid}", errors=errors)
    if d is None:
        return None
    return [{"name": f.get("key"), "size": f.get("size")} for f in d.get("files", [])]


def verify_accession(acc: Accession, *, errors: list | None = None) -> Accession:
    """Fetch the file listing for GEO/Zenodo accessions. `verified` means the listing was OPENED —
    an empty listing is a verified fact about the deposit, not a failure."""
    repo = acc.repository.lower()
    if repo == "geo" and re.match(r"^GSE\d+$", acc.id, re.I):
        files = verify_geo(acc.id, errors=errors)
        url = f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={acc.id.upper()}"
    elif repo == "zenodo":
        files = verify_zenodo(acc.id, errors=errors)
        url = f"https://zenodo.org/records/{re.sub(r'[^0-9]', '', acc.id.split('zenodo.')[-1])}"
    else:
        return acc
    if files is None:
        # Listing unavailable right now (throttled/offline). Keep whatever an earlier check actually
        # saw instead of erasing it — a transient 403 must not turn a verified deposit into a guess.
        return acc.model_copy(update={"url": acc.url or url})
    return acc.model_copy(update={"files": files, "verified": True, "url": acc.url or url})


def has_table(acc: Accession) -> bool:
    return any(TABLE_EXT.search(f.get("name") or "") for f in acc.files)


def verify_entry(entry, *, errors: list | None = None):
    """Verify every GEO/Zenodo accession of a CatalogueEntry in place and set its verification:
    VERIFIED (a per-sequence-style table is listed) | DEPOSIT_ONLY (deposit exists, no table seen) |
    UNVERIFIED (nothing could be checked)."""
    entry.accessions = [verify_accession(a, errors=errors) for a in entry.accessions]
    # Only the dataset's OWN deposits may grant verification: a `mentioned` accession was text-mined
    # from the full text and usually belongs to a cited dataset, so listing its files says nothing
    # about whether THIS dataset is obtainable.
    own = [a for a in entry.accessions if a.relation == "deposit"]
    entry.verification = ("VERIFIED" if any(a.verified and has_table(a) for a in own)
                          else "DEPOSIT_ONLY" if any(a.verified for a in own) else "UNVERIFIED")
    return entry
