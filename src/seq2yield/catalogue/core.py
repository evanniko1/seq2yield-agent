"""Dataset catalogue — source of truth + dedupe + registry sync.

Truth lives in `data/catalogue/datasets.jsonl` (curated, diffable, one entry per line, sorted by id).
`orchestration.store.ingest_catalogue` projects it into the project SQLite read-model so the
dashboard, the dataset registry and onboarding see one catalogue.

Discovery rule (read-literature): `add()` SKIPS a candidate whose publication (DOI / PMID / title) or
dataset (accession / id) is already recorded, and ADDS it only if it is entirely new. Lifecycle changes
(candidate -> scheduled -> registered, verification) go through explicit curation calls instead.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from .model import Accession, CatalogueEntry, Publication

ROOT = Path(__file__).resolve().parents[3]
CATALOGUE_PATH = ROOT / "data" / "catalogue" / "datasets.jsonl"


# ------------------------------------------------------------------------------ normalisation ---
def norm_doi(doi: str | None) -> str | None:
    if not doi:
        return None
    d = doi.strip().lower()
    d = re.sub(r"^(https?://(dx\.)?doi\.org/|doi:\s*)", "", d)
    return d or None


def norm_pmid(pmid: str | int | None) -> str | None:
    if pmid is None:
        return None
    s = re.sub(r"\D", "", str(pmid))
    return s or None


def norm_title(title: str | None) -> str | None:
    if not title:
        return None
    t = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode().lower()
    t = re.sub(r"[^a-z0-9]+", " ", t).strip()
    return t or None


def norm_accession(repo: str | None, acc: str | None) -> str | None:
    """Repository-agnostic key: GSE/SRP/PRJNA/ENCSR... uppercased; Zenodo as ZENODO:<record>."""
    if not acc:
        return None
    a = acc.strip()
    m = re.search(r"zenodo\.(?:org/records?/)?(\d+)", a, re.I)      # 10.5281/zenodo.N | zenodo.org/records/N
    if m:
        return f"ZENODO:{m.group(1)}"
    if (repo or "").lower() == "zenodo" and re.fullmatch(r"\d+", a):
        return f"ZENODO:{a}"
    return a.upper()


def _title_tokens(t: str) -> set[str]:
    return {w for w in t.split() if len(w) > 2}


def _titles_match(a: str, b: str) -> bool:
    """Same work under a different title? Exact; near-identical (Jaccard >= 0.9); or the preprint vs
    published pattern — one title is a word-prefix of the other ("...elements" vs "...elements in three
    cell types"), or the shorter title's words are all contained in the longer one (published titles
    often drop clauses). Containment needs >= 8 content words so short generic titles never merge."""
    if a == b:
        return True
    ta, tb = _title_tokens(a), _title_tokens(b)
    if min(len(ta), len(tb)) < 6:                  # short titles: exact only (avoid false merges)
        return False
    if len(ta & tb) / len(ta | tb) >= 0.9:
        return True
    sa, sb = a.split(), b.split()
    short, longer = (sa, sb) if len(sa) <= len(sb) else (sb, sa)
    if len(short) >= 6 and longer[:len(short)] == short:
        return True
    small, big = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    return len(small) >= 8 and small <= big


# ---------------------------------------------------------------------------------- storage ---
def load(path: Path | None = None) -> list[CatalogueEntry]:
    p = Path(path or CATALOGUE_PATH)
    if not p.exists():
        return []
    return [CatalogueEntry.model_validate_json(ln)
            for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


def save(entries: list[CatalogueEntry], path: Path | None = None) -> Path:
    """Atomic write, sorted by id (stable diffs). Refuses duplicate ids."""
    p = Path(path or CATALOGUE_PATH)
    ids = [e.id for e in entries]
    dup = {i for i in ids if ids.count(i) > 1}
    if dup:
        raise ValueError(f"duplicate catalogue ids: {sorted(dup)}")
    p.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(e.model_dump_json(exclude_none=False) + "\n"
                   for e in sorted(entries, key=lambda e: e.id))
    fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=".catalogue-", suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    os.replace(tmp, p)
    return p


# ----------------------------------------------------------------------------------- dedupe ---
def find_match(entries: list[CatalogueEntry], cand: CatalogueEntry) -> tuple[str, str] | None:
    """(existing_id, matched_on) if any publication or accession of `cand` is already recorded."""
    for e in entries:
        for cp in cand.publications:
            for ep in e.publications:
                if norm_doi(cp.doi) and norm_doi(cp.doi) == norm_doi(ep.doi):
                    return e.id, f"doi:{norm_doi(cp.doi)}"
                if norm_pmid(cp.pmid) and norm_pmid(cp.pmid) == norm_pmid(ep.pmid):
                    return e.id, f"pmid:{norm_pmid(cp.pmid)}"
    for e in entries:
        keys = {norm_accession(a.repository, a.id) for a in e.accessions if a.relation == "deposit"}
        for ca in cand.accessions:
            if ca.relation != "deposit":
                continue
            k = norm_accession(ca.repository, ca.id)
            if k and k in keys:
                return e.id, f"accession:{k}"
    for e in entries:
        for cp in cand.publications:
            ct = norm_title(cp.title)
            if not ct:
                continue
            for ep in e.publications:
                et = norm_title(ep.title)
                if et and _titles_match(ct, et):
                    return e.id, "title"
    for e in entries:
        if cand.dataset_spec and cand.dataset_spec == e.dataset_spec:
            return e.id, f"dataset_spec:{cand.dataset_spec}"
    # An id is a human-readable slug (firstauthor_year), NOT an identity: two different datasets can
    # share one (two Zhang 2025 papers). Matching on it would silently drop the second. So it is a
    # key only when the candidate offers nothing better; `add` renames a colliding slug instead.
    if not cand.publications and not cand.accessions:
        for e in entries:
            if e.id == cand.id:
                return e.id, "id"
    return None


@dataclass
class AddResult:
    action: str                    # "added" | "skipped"
    id: str
    matched_on: str | None = None  # why it was skipped


def _unique_id(entries: list[CatalogueEntry], base: str) -> str:
    taken = {e.id for e in entries}
    if base not in taken:
        return base
    for suf in "bcdefghijklmnopqrstuvwxyz":
        if f"{base}{suf}" not in taken:
            return f"{base}{suf}"
    raise ValueError(f"cannot allocate an id for {base}")


def suggest_id(cand: CatalogueEntry) -> str:
    p = cand.primary
    if p and p.citation:
        head = p.citation.strip().split(",")[0]
        first = re.sub(r"\s+et\s+al\.?$", "", head)    # keep particles: "de Boer" -> "deboer"
        first = unicodedata.normalize("NFKD", first).encode("ascii", "ignore").decode().lower()
        first = re.sub(r"[^a-z]", "", first) or "unknown"
        return f"{first}_{p.year}" if p.year else first
    return cand.id or "unknown"


def add(entries: list[CatalogueEntry], cand: CatalogueEntry) -> AddResult:
    """Skip if any publication/accession is already recorded; otherwise append (mutates `entries`)."""
    m = find_match(entries, cand)
    if m:
        return AddResult("skipped", m[0], m[1])
    if not cand.id or cand.id in {e.id for e in entries}:
        cand = cand.model_copy(update={"id": _unique_id(entries, cand.id or suggest_id(cand))})
    entries.append(cand)
    return AddResult("added", cand.id)


# ------------------------------------------------------------------------- curation (explicit) ---
def get(entries: list[CatalogueEntry], entry_id: str) -> CatalogueEntry:
    for e in entries:
        if e.id == entry_id:
            return e
    raise KeyError(entry_id)


def attach_publication(entries, entry_id: str, pub: Publication) -> bool:
    """Link an extra publication to an existing dataset (e.g. a benchmark paper). False if present."""
    e = get(entries, entry_id)
    probe = CatalogueEntry(id="__probe__", publications=[pub])
    hit = find_match([e], probe)
    if hit:
        return False
    e.publications.append(pub)
    return True


def attach_accession(entries, entry_id: str, acc: Accession) -> bool:
    e = get(entries, entry_id)
    k = norm_accession(acc.repository, acc.id)
    if k in {norm_accession(a.repository, a.id) for a in e.accessions}:
        return False
    e.accessions.append(acc)
    return True


def set_status(entries, entry_id: str, status: str, *, reason: str | None = None) -> CatalogueEntry:
    e = get(entries, entry_id)
    CatalogueEntry.model_validate({**e.model_dump(), "status": status})   # validate before mutating
    e.status = status
    if reason:
        e.notes = (f"{e.notes} | " if e.notes else "") + reason
    return e


# The registry stores coarse labels ("yeast", "utr"); a candidate carries a precise one
# ("Saccharomyces cerevisiae", "utr5"). Scoring must compare like with like, or every human cell
# line looks like a new organism and every 5'UTR library like a new modality.
_ORG_ALIAS = {"yeast": "saccharomyces cerevisiae", "ecoli": "escherichia coli",
              "human": "homo sapiens", "mouse": "mus musculus"}
_MOD_ALIAS = {"utr": "utr5"}        # both registered "utr" datasets are 5'UTR libraries


def _species(organism: str | None) -> set[str]:
    """Species mentioned by an organism label, without cell-line/context parentheticals."""
    out = set()
    for part in (organism or "").split(","):
        name = re.sub(r"\(.*?\)|\+", " ", part).strip().lower()
        name = re.sub(r"\s+", " ", name)
        if name:
            out.add(_ORG_ALIAS.get(name, name))
    return out


def _modality(modality: str | None) -> str:
    m = (modality or "").strip().lower()
    return _MOD_ALIAS.get(m, m)


def score(e: CatalogueEntry, entries: list[CatalogueEntry]) -> int:
    """Transparent project-value score (read-literature Step 4)."""
    reg = [x for x in entries if x.status == "registered" and x.id != e.id]
    orgs = {sp for x in reg for sp in _species(x.organism)}
    mods = {_modality(x.modality) for x in reg}
    new_org = bool(_species(e.organism)) and not (_species(e.organism) & orgs)
    new_mod = bool(_modality(e.modality)) and _modality(e.modality) not in mods
    s = 0
    if new_org:
        s += 2
    if new_mod:
        s += 2
    if new_org and not new_mod:
        s += 1                                          # transfer pair: same element, new organism
    s += 1 if e.hts_readout else 0
    s += 1 if e.provided_splits else 0
    s += 1 if (e.replicates or 0) >= 2 else 0
    if e.accessions and all(a.repository.upper() in ("SRA", "ENA") for a in e.accessions):
        s -= 2                                          # raw reads only
    if (e.library_type or "").startswith("variant"):
        s -= 1
    if e.access == "controlled":
        s -= 3
    return s


# ------------------------------------------------------------------- registry synchronisation ---
_ORG = {"ecoli": "Escherichia coli", "yeast": "Saccharomyces cerevisiae", "human": "Homo sapiens"}


def _spec_publications(citation: str) -> list[Publication]:
    pubs = []
    for part in [p.strip() for p in (citation or "").split(";") if p.strip()]:
        if re.search(r"zenodo", part, re.I) and not re.search(r"et al", part):
            continue                                   # an accession mention, not a publication
        y = re.search(r"\((\d{4})\)", part)
        doi = None
        m = re.search(r"bioRxiv\s+(\d{4}\.\d{2}\.\d{2}\.\d+)", part)
        if m:
            doi = f"10.1101/{m.group(1)}"              # bioRxiv DOIs are 10.1101/<id> by construction
        pubs.append(Publication(citation=part, year=int(y.group(1)) if y else None, doi=doi,
                                role="primary" if not pubs else "related"))
    return pubs


def _spec_accessions(source: dict) -> list[Accession]:
    out = []
    for key, repo in (("geo", "GEO"), ("zenodo", "Zenodo"), ("repo", "GitHub"), ("portal", "Portal")):
        v = (source or {}).get(key)
        if v:
            out.append(Accession(repository=repo, id=str(v),
                                 url=v if str(v).startswith("http") else None))
    return out


def sync_registry(entries: list[CatalogueEntry]) -> list[str]:
    """Make every DatasetSpec in configs/datasets/ a `registered` entry (create or promote).
    Returns human-readable change notes. Never demotes."""
    from ..data import datasets
    changes = []
    for did in datasets.all_ids():
        s = datasets.spec(did)
        existing = next((e for e in entries if e.dataset_spec == did or e.id == did), None)
        if existing:
            if existing.status != "registered" or existing.dataset_spec != did:
                existing.status, existing.dataset_spec = "registered", did
                changes.append(f"promoted {existing.id} -> registered ({did})")
            continue
        e = CatalogueEntry(
            id=did, title=s.display_name or did, publications=_spec_publications(s.citation),
            accessions=_spec_accessions(s.source), organism=_ORG.get(s.organism, s.organism),
            modality=s.modality, seq_len_nt=s.seq_len, fixed_length=True,
            library_type=None, continuous=True, status="registered", fit_verdict="STRONG_FIT",
            dataset_spec=did, discovered_by="seed:registry")
        # if a publication of this spec is already catalogued under another id, link instead
        hit = find_match(entries, e)
        if hit:
            other = get(entries, hit[0])
            other.status, other.dataset_spec = "registered", did
            changes.append(f"linked {other.id} -> registered ({did}) via {hit[1]}")
            continue
        entries.append(e)
        changes.append(f"added {did} (registered)")
    return changes
