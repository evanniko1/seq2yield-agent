"""The dataset catalogue: dedupe (publication- and deposit-level), add/skip, atomic persistence,
registry sync, project-value scoring, the network resolvers (all mocked — no live calls in tests),
the SQLite projection, and the dashboard page's escaping of web-sourced text.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seq2yield import catalogue as cat  # noqa: E402
from seq2yield.catalogue import Accession, CatalogueEntry, Publication, resolve  # noqa: E402


def _entry(eid, *, doi=None, pmid=None, title=None, accs=(), **kw):
    return CatalogueEntry(
        id=eid, publications=[Publication(citation=eid, doi=doi, pmid=pmid, title=title)],
        accessions=[Accession(repository=r, id=a, relation=rel) for r, a, rel in accs], **kw)


# ----------------------------------------------------------------- normalisation + title matching
def test_accession_normalisation_handles_zenodo_shapes():
    assert cat.norm_accession("GEO", "gse114002") == "GSE114002"
    assert cat.norm_accession("Zenodo", "10.5281/zenodo.7395397") == "ZENODO:7395397"
    assert cat.norm_accession("Zenodo", "https://zenodo.org/records/7395397") == "ZENODO:7395397"
    assert cat.norm_accession("Zenodo", "7395397") == "ZENODO:7395397"


def test_preprint_and_published_titles_are_the_same_publication():
    """A preprint keeps its own DOI, so only the title can tie it to the published paper."""
    pub = "Massively parallel characterization of transcriptional regulatory elements"
    pre = pub + " in three diverse human cell types"
    entries = [_entry("agarwal_2025", doi="10.1038/s41586-025-1", title=pub)]
    assert cat.find_match(entries, _entry("x", doi="10.1101/2023.03.05.531189", title=pre))


def test_unrelated_titles_do_not_match():
    entries = [_entry("a", title="Deciphering the rules by which 5'UTR sequences affect expression")]
    assert cat.find_match(entries, _entry("b", title="Systematic dissection of regulatory motifs")) is None


# ----------------------------------------------------------------------------- dedupe keys
def test_doi_pmid_and_deposit_accession_are_dedupe_keys():
    entries = [_entry("known", doi="10.1/a", pmid="111", accs=[("GEO", "GSE1", "deposit")])]
    assert cat.find_match(entries, _entry("c", doi="10.1/A"))[1].startswith("doi:")
    assert cat.find_match(entries, _entry("c", pmid="111"))[1].startswith("pmid:")
    assert cat.find_match(entries, _entry("c", accs=[("GEO", "gse1", "deposit")]))[1].startswith("accession:")


def test_mentioned_accessions_are_not_dedupe_keys():
    """A text-mined accession usually belongs to a dataset the paper CITES. Using it as a key would
    silently skip that cited dataset when it is discovered later on its own."""
    entries = [_entry("citing_paper", doi="10.1/x", accs=[("GEO", "GSE83120", "mentioned")])]
    cand = _entry("the_cited_dataset", doi="10.1/y", accs=[("GEO", "GSE83120", "deposit")])
    assert cat.find_match(entries, cand) is None


# ------------------------------------------------------------------------------- add / skip
def test_add_skips_recorded_and_adds_new():
    entries = [_entry("known", doi="10.1/a")]
    assert cat.add(entries, _entry("dup", doi="10.1/a")).action == "skipped"
    assert cat.add(entries, _entry("fresh", doi="10.1/b")).action == "added"
    assert len(entries) == 2 and "fresh" in {e.id for e in entries}


def test_two_different_datasets_may_share_a_slug():
    """firstauthor_year is a label, not an identity: the second Zhang 2025 dataset must still be
    added (under a fresh id), not skipped as a duplicate of the first."""
    entries = [_entry("zhang_2025", doi="10.1/a", title="Cross-species bacterial promoters")]
    res = cat.add(entries, _entry("zhang_2025", doi="10.1/b", title="Evaluation of MPRA designs"))
    assert res.action == "added" and res.id != "zhang_2025" and len(entries) == 2


def test_an_entry_with_no_identifiers_falls_back_to_its_id():
    entries = [CatalogueEntry(id="align_seq2expr", status="watch")]
    assert cat.add(entries, CatalogueEntry(id="align_seq2expr")).action == "skipped"


# --------------------------------------------------------------------- persistence + lifecycle
def test_save_load_roundtrip_is_sorted_and_atomic(tmp_path):
    p = tmp_path / "c.jsonl"
    cat.save([_entry("b", doi="10.1/b"), _entry("a", doi="10.1/a")], p)
    assert [e.id for e in cat.load(p)] == ["a", "b"]
    assert not list(tmp_path.glob("*.tmp*"))          # temp file replaced, nothing left behind


def test_save_refuses_duplicate_ids(tmp_path):
    with pytest.raises(ValueError):
        cat.save([_entry("a", doi="10.1/a"), _entry("a", doi="10.1/b")], tmp_path / "c.jsonl")


def test_load_of_a_missing_file_is_empty(tmp_path):
    assert cat.load(tmp_path / "nope.jsonl") == []


def test_set_status_validates_before_mutating():
    entries = [_entry("a", doi="10.1/a")]
    cat.set_status(entries, "a", "scheduled", reason="vetted")
    assert entries[0].status == "scheduled" and "vetted" in (entries[0].notes or "")
    with pytest.raises(ValueError):
        cat.set_status(entries, "a", "promoted")
    assert entries[0].status == "scheduled"           # unchanged by the rejected transition


def test_sync_registry_marks_specs_registered_without_demoting(tmp_path):
    entries = []
    cat.sync_registry(entries)
    reg = {e.id for e in entries if e.status == "registered"}
    assert {"ecoli", "yeast"} <= reg                  # every configs/datasets/*.yaml is represented
    e = cat.get(entries, "yeast")
    e.status, e.notes = "registered", None
    cat.sync_registry(entries)                        # idempotent: no duplicates, no status churn
    assert len([x for x in entries if x.id == "yeast"]) == 1


# ------------------------------------------------------------------------------- scoring
def test_score_compares_species_not_label_text():
    """"Homo sapiens (K562)" is not a new organism when Homo sapiens is already registered."""
    reg = [_entry("human_reg", organism="human", modality="regulatory", status="registered"),
           _entry("yeast_reg", organism="yeast", modality="promoter", status="registered")]
    same = _entry("c1", organism="Homo sapiens (K562)", modality="regulatory", hts_readout=True)
    new_org = _entry("c2", organism="Danio rerio (embryos)", modality="regulatory", hts_readout=True)
    assert cat.score(same, reg + [same]) == 1                       # hts only
    assert cat.score(new_org, reg + [new_org]) == 4                 # +2 new organism, +1 transfer pair, +1 hts


def test_score_rewards_a_new_modality_and_penalises_raw_reads_only():
    reg = [_entry("human_reg", organism="human", modality="regulatory", status="registered")]
    new_mod = _entry("c", organism="Homo sapiens", modality="terminator", hts_readout=True)
    assert cat.score(new_mod, reg + [new_mod]) == 3                 # +2 new modality, +1 hts
    raw = _entry("r", organism="Homo sapiens", modality="regulatory", hts_readout=True,
                 accs=[("SRA", "SRP1", "deposit")])
    assert cat.score(raw, reg + [raw]) == -1                        # +1 hts, -2 raw reads only


# --------------------------------------------------------------- resolvers (network mocked)
def test_build_query_phrases_multiword_terms_and_keeps_wildcards():
    q = resolve.build_query([["massively parallel reporter assay", "MPRA"], ["random*"]])
    assert '(TITLE:"massively parallel reporter assay" OR ABSTRACT:"massively parallel reporter assay")' in q
    assert "(TITLE:random* OR ABSTRACT:random*)" in q               # wildcards must stay unquoted
    assert q.count(" AND ") == 1 and q.count(" OR ") >= 2


def test_epmc_search_all_follows_cursor_marks(monkeypatch):
    pages = {"*": {"hitCount": 3, "nextCursorMark": "c2", "resultList": {"result": [{"id": "1"}, {"id": "2"}]}},
             "c2": {"hitCount": 3, "nextCursorMark": "c2", "resultList": {"result": [{"id": "3"}]}}}

    def fake_get(url, **kw):
        import urllib.parse
        mark = urllib.parse.unquote(url.split("cursorMark=")[1].split("&")[0])
        return pages[mark]

    monkeypatch.setattr(resolve, "_get", fake_get)
    rows, total = resolve.epmc_search_all("q", max_hits=10)
    assert total == 3 and [r["id"] for r in rows] == ["1", "2", "3"]


def test_citation_resolution_refuses_an_ambiguous_match(monkeypatch):
    """Two different Rosenberg papers: without positive evidence the resolver must return None
    rather than take the first hit."""
    hits = [{"doi": "10.1/aaa", "title": "ATPase structure", "authorString": "Rosenberg OS, et al.",
             "journalTitle": "Cell", "pubYear": "2015", "journalVolume": "163", "pageInfo": "1-10"},
            {"doi": "10.1/bbb", "title": "Learning the sequence determinants of splicing",
             "authorString": "Rosenberg AB, et al.", "journalTitle": "Cell", "pubYear": "2015",
             "journalVolume": "163", "pageInfo": "698-711"}]
    monkeypatch.setattr(resolve, "epmc_search", lambda *a, **k: hits)
    assert resolve.resolve_citation("Rosenberg et al., Cell (2015)") is None
    got = resolve.resolve_citation("Rosenberg et al., Cell 163:698-711 (2015)")
    assert got is not None and got.doi == "10.1/bbb"                # volume:first-page is the evidence


def test_text_mined_accessions_keep_only_deposits_and_are_marked_mentioned(monkeypatch):
    """The annotations API returns rsIDs and prose alongside deposits; only repository accessions
    are kept, and they are `mentioned` until ownership is established."""
    monkeypatch.setattr(resolve, "_get", lambda url, **kw: [{
        "annotations": [{"exact": "GSE12345", "type": "Accession Numbers"},
                        {"exact": "rs7903146", "type": "Accession Numbers"},
                        {"exact": "E-MTAB-7144", "type": "Accession Numbers"},
                        {"exact": "not-an-accession", "type": "Accession Numbers"}]}])
    accs = resolve.text_mined_accessions("PMC7777777")
    assert [(a.repository, a.id, a.relation) for a in accs] == [
        ("GEO", "GSE12345", "mentioned"), ("ArrayExpress", "E-MTAB-7144", "mentioned")]
    assert resolve.text_mined_accessions(None) == []


def test_geo_deposits_need_the_publication_to_own_them(monkeypatch):
    pub = Publication(citation="x", pmid="42")
    monkeypatch.setattr(resolve, "text_mined_accessions", lambda p, **k: [
        Accession(repository="GEO", id="GSE_OWN", relation="mentioned"),
        Accession(repository="GEO", id="GSE_CITED", relation="mentioned")])
    monkeypatch.setattr(resolve, "geo_pubmed_ids", lambda g, **k: ["42"] if g == "GSE_OWN" else ["777"])
    out = {a.id: a.relation for a in resolve.own_deposits(pub)}
    assert out == {"GSE_OWN": "deposit", "GSE_CITED": "mentioned"}


def test_verify_geo_separates_unreadable_from_empty(monkeypatch):
    monkeypatch.setattr(resolve, "_get", lambda url, **kw: None)
    assert resolve.verify_geo("GSE1") is None                       # could not be read
    empty = '<pre><a href="/geo/series/GSE1nnn/GSE1/">Parent Directory</a>   -   </pre>'
    monkeypatch.setattr(resolve, "_get", lambda url, **kw: empty)
    assert resolve.verify_geo("GSE1") == []                         # read, and genuinely empty


def test_verification_label_only_counts_own_deposits(monkeypatch):
    monkeypatch.setattr(resolve, "verify_accession", lambda a, **k: a.model_copy(
        update={"verified": True, "files": [{"name": "table.csv"}]}))
    mentioned_only = _entry("m", doi="10.1/a", accs=[("Zenodo", "1", "mentioned")])
    own = _entry("o", doi="10.1/b", accs=[("Zenodo", "2", "deposit")])
    assert resolve.verify_entry(mentioned_only).verification == "UNVERIFIED"
    assert resolve.verify_entry(own).verification == "VERIFIED"


def test_verification_downgrades_when_no_table_is_listed(monkeypatch):
    monkeypatch.setattr(resolve, "verify_accession",
                        lambda a, **k: a.model_copy(update={"verified": True, "files": [{"name": "raw.tar"}]}))
    e = _entry("o", doi="10.1/b", accs=[("GEO", "GSE1", "deposit")])
    e.verification = "VERIFIED"
    assert resolve.verify_entry(e).verification == "DEPOSIT_ONLY"


# ------------------------------------------------------- projection into the SQLite read-model
def _catalogue_file(tmp_path: Path) -> Path:
    p = tmp_path / "data" / "catalogue"
    p.mkdir(parents=True)
    f = p / "datasets.jsonl"
    cat.save([
        CatalogueEntry(id="d1", title="A dataset", organism="Homo sapiens", modality="promoter",
                       status="candidate", fit_verdict="STRONG_FIT", project_value={"score": 4},
                       publications=[Publication(citation="Smith et al. (2025)", doi="10.1/a",
                                                 title="T", year=2025, role="primary"),
                                     Publication(citation="Smith et al. (2024)", doi="10.1101/x",
                                                 role="preprint")],
                       accessions=[Accession(repository="GEO", id="GSE1", relation="deposit",
                                             verified=True, files=[{"name": "t.csv"}]),
                                   Accession(repository="Zenodo", id="9", relation="mentioned")]),
        CatalogueEntry(id="d2", title="Rejected", status="rejected", fit_verdict="REJECT",
                       reject_reason="too small", organism="Homo sapiens", modality="promoter"),
    ], f)
    return f


def test_ingest_catalogue_is_idempotent(tmp_path):
    from orchestration import store
    f = _catalogue_file(tmp_path)
    con = store.connect(tmp_path / "db.sqlite")
    for _ in range(2):
        assert store.ingest_catalogue(con, f) == 2
    con.commit()
    assert con.execute("SELECT COUNT(*) FROM catalogue").fetchone()[0] == 2
    assert con.execute("SELECT COUNT(*) FROM catalogue_publication").fetchone()[0] == 2
    assert con.execute("SELECT COUNT(*) FROM catalogue_accession").fetchone()[0] == 2
    row = store.catalogue_table(con, "candidate")[0]
    assert row["id"] == "d1" and row["n_deposits"] == 1 and row["doi"] == "10.1/a"
    cov = store.catalogue_coverage(con)
    assert cov["by_status"] == {"candidate": 1, "rejected": 1}
    assert [c["status"] for c in cov["cells"]] == ["candidate"]      # rejects stay out of the gap map


def test_ingest_catalogue_drops_records_removed_from_the_file(tmp_path):
    from orchestration import store
    f = _catalogue_file(tmp_path)
    con = store.connect(tmp_path / "db.sqlite")
    store.ingest_catalogue(con, f)
    entries = cat.load(f)
    cat.get(entries, "d1").publications.pop()                        # drop the preprint
    cat.save(entries, f)
    store.ingest_catalogue(con, f)
    assert con.execute("SELECT COUNT(*) FROM catalogue_publication").fetchone()[0] == 1


# ------------------------------------------------------------------------- dashboard page
def test_catalogue_page_escapes_publication_text(tmp_path, monkeypatch):
    import importlib.util
    spec = importlib.util.spec_from_file_location("dash_cat", ROOT / "scripts" / "run_dashboard.py")
    dash = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dash)

    from orchestration import store
    db = tmp_path / "db.sqlite"
    con = store.connect(db)
    nasty = '<img src=x onerror=alert(1)>'
    cat.save([CatalogueEntry(id="x1", title=nasty, status="candidate", organism="Homo sapiens",
                             modality="promoter", project_value={"score": 1},
                             publications=[Publication(citation=nasty, doi="javascript:alert(1)")])],
             tmp_path / "c.jsonl")
    store.ingest_catalogue(con, tmp_path / "c.jsonl")
    con.commit()
    con.close()
    monkeypatch.setattr(dash, "_con", lambda: store.connect(db))
    html = dash.app.test_client().get("/catalogue").get_data(as_text=True)
    assert nasty not in html and "&lt;img" in html                   # rendered as text, not markup
    assert "href='javascript" not in html                            # a non-DOI is never linkified
    assert "x1" in html


def test_council_pages_escape_llm_text(tmp_path, monkeypatch):
    """The dashboard body is rendered with |safe, so Jinja escapes nothing: every page that prints
    LLM prose (proposal hypotheses, reviewer reject reasons, claims) must escape it itself, and the
    path parameter must not be reflected raw."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("dash_council", ROOT / "scripts" / "run_dashboard.py")
    dash = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dash)

    from orchestration import store
    db = tmp_path / "db.sqlite"
    con = store.connect(db)
    hyp = "<script>alert('hyp')</script>"
    rej = "<img src=x onerror=alert('rej')>"
    claim = "<svg onload=alert('claim')>"
    con.execute("INSERT OR REPLACE INTO council_query VALUES ('t1','ts','f','r',1,'approve','H1',NULL)")
    con.execute("INSERT OR REPLACE INTO proposal VALUES ('t1','H1','cnn','rf','model_architecture',"
                "'ecoli','all','t',?)", (hyp,))
    con.execute("INSERT OR REPLACE INTO review VALUES ('t1','H1','critic',3,3,3,3,?)", (rej,))
    con.execute("INSERT OR REPLACE INTO claim (run_id,ts,candidate_model,baseline_model,"
                "mean_delta_r2,ci_excludes_zero,p_value,bootstrap_unit,dataset,train_size,status,"
                "claim,baseline_source) VALUES ('R1','ts','cnn','rf',0.03,1,0.01,'series',"
                "'ecoli',2000,'accepted',?,'registry:x')", (claim,))
    con.commit()
    con.close()
    monkeypatch.setattr(dash, "_con", lambda: store.connect(db))   # each route closes its own
    c = dash.app.test_client()

    detail = c.get("/query/t1").get_data(as_text=True)
    assert hyp not in detail and rej not in detail
    assert "&lt;script&gt;" in detail and "&lt;img" in detail
    home = c.get("/").get_data(as_text=True)
    assert claim not in home and "&lt;svg" in home
    reflected = c.get("/query/<img src=x onerror=alert(1)>").get_data(as_text=True)
    assert "<img src=x" not in reflected and "&lt;img src=x" in reflected
    assert c.get("/dataset/no-such-dataset").status_code == 404      # unknown id, not a 500


def test_catalogue_page_ignores_an_unknown_status_filter(tmp_path, monkeypatch):
    import importlib.util
    spec = importlib.util.spec_from_file_location("dash_cat2", ROOT / "scripts" / "run_dashboard.py")
    dash = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dash)

    from orchestration import store
    db = tmp_path / "db.sqlite"
    con = store.connect(db)
    store.ingest_catalogue(con, _catalogue_file(tmp_path))
    con.commit()
    con.close()
    monkeypatch.setattr(dash, "_con", lambda: store.connect(db))
    c = dash.app.test_client()
    bogus = c.get("/catalogue?status=<script>")
    assert bogus.status_code == 200 and "d1" in bogus.get_data(as_text=True)
    narrowed = c.get("/catalogue?status=registered")
    assert narrowed.status_code == 200 and "d1" not in narrowed.get_data(as_text=True)


# --------------------------------------------------------------- the real catalogue file is sane
def test_shipped_catalogue_is_valid_and_has_no_duplicate_publications():
    entries = cat.load()
    assert entries, "data/catalogue/datasets.jsonl should not be empty"
    ids = [e.id for e in entries]
    assert len(ids) == len(set(ids))
    seen_doi: dict[str, str] = {}
    for e in entries:
        for p in e.publications:
            if p.doi:
                assert p.doi.lower() not in seen_doi, f"{p.doi} in {e.id} and {seen_doi.get(p.doi.lower())}"
                seen_doi[p.doi.lower()] = e.id
        for a in e.accessions:
            assert a.relation in ("deposit", "mentioned")
        if e.status == "rejected":
            assert e.reject_reason, f"{e.id} is rejected without a reason"
        else:
            assert e.fit_verdict or e.status == "watch"


def test_shipped_catalogue_registered_entries_track_the_registry():
    entries = cat.load()
    specs = {p.stem for p in (ROOT / "configs" / "datasets").glob("*.yaml")}
    registered = {e.id for e in entries if e.status == "registered"}
    assert specs <= registered, f"registry drift: {specs - registered} missing from the catalogue"
    for e in entries:
        if e.id in specs:
            assert e.dataset_spec, f"{e.id} should name its DatasetSpec file"
