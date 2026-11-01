"""Dataset catalogue records.

One `CatalogueEntry` = one DATASET, which may be described by several PUBLICATIONS (e.g. dream2022 =
de Boer 2020 + the DREAM 2022 challenge) and deposited under several ACCESSIONS (GEO + Zenodo + a
code repo). Dedupe works at both levels: a candidate whose publication or accession is already
recorded is a duplicate, even if it arrives under a different name.
"""
from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field, field_validator

STATUSES = ("registered", "scheduled", "candidate", "watch", "rejected")
#   registered — a DatasetSpec exists in configs/datasets/ (the council can target it when data is present)
#   scheduled  — vetted strong fit, queued for onboarding
#   candidate  — discovered, triaged, not yet scheduled
#   watch      — relevant but not yet obtainable (embargoed / forthcoming / access pending)
#   rejected   — fails the project fit (kept so it is never re-added as "new")
VERDICTS = ("STRONG_FIT", "FIT_WITH_CAVEATS", "REJECT")


class Publication(BaseModel):
    citation: str                                  # "Sample et al., Nat Biotechnol (2019)"
    doi: str | None = None
    pmid: str | None = None
    pmcid: str | None = None
    title: str | None = None
    year: int | None = None
    role: str = "primary"                          # primary | benchmark | reanalysis | related


class Accession(BaseModel):
    repository: str                                # GEO | Zenodo | SRA | ENA | ENCODE | Synapse | GitHub | ...
    id: str                                        # GSE114002 | 7395397 | ENCSR... | owner/repo
    url: str | None = None
    files: list[dict] = Field(default_factory=list)   # [{"name": ..., "size": ...}]
    verified: bool = False                         # True only after the file listing was seen
    relation: str = "deposit"                      # deposit (this dataset's own) | mentioned (cited/text-mined)
    #   Only `deposit` accessions are dedupe keys: a paper's full text also cites OTHER datasets, and
    #   using those would wrongly skip the cited dataset when it is discovered later.


class CatalogueEntry(BaseModel):
    id: str                                        # DatasetSpec id when registered, else firstauthor_year
    title: str | None = None                       # short human description of the dataset
    publications: list[Publication] = Field(default_factory=list)
    accessions: list[Accession] = Field(default_factory=list)
    organism: str | None = None
    cell_context: str | None = None
    modality: str | None = None                    # promoter|enhancer|regulatory|utr5|utr3|rbs|coding|...
    assay: str | None = None                       # MPRA|lentiMPRA|STARR-seq|Sort-seq|FACS-seq|polysome-MPRA|GPRA|...
    hts_readout: bool | None = None                # per-oligo readout counted by sequencing
    readout: str | None = None                     # exactly what was measured
    continuous: bool | None = None
    n_sequences: int | None = None
    seq_len_nt: int | None = None
    fixed_length: bool | None = None
    library_type: str | None = None                # random|designed|genomic tiles|variant pairs|saturation mutagenesis
    replicates: int | None = None
    provided_splits: bool | None = None
    access: str | None = None                      # open|login|controlled|on-request|embargoed
    license: str | None = None
    status: str = "candidate"
    fit_verdict: str | None = None
    caveats: list[str] = Field(default_factory=list)
    reject_reason: str | None = None
    project_value: dict = Field(default_factory=dict)
    dataset_spec: str | None = None                # configs/datasets/<id>.yaml when registered
    backlog_ref: str | None = None
    evidence: list[str] = Field(default_factory=list)   # URLs actually opened
    discovered_by: str = "manual"                  # seed:registry | seed:docs/BACKLOG.md | read-literature:<date>
    added: str = Field(default_factory=lambda: date.today().isoformat())
    verification: str = "UNVERIFIED"               # VERIFIED (table seen) | DEPOSIT_ONLY | UNVERIFIED
    notes: str | None = None

    @field_validator("status")
    @classmethod
    def _status(cls, v: str) -> str:
        if v not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}, got {v!r}")
        return v

    @field_validator("fit_verdict")
    @classmethod
    def _verdict(cls, v: str | None) -> str | None:
        if v is not None and v not in VERDICTS:
            raise ValueError(f"fit_verdict must be one of {VERDICTS}, got {v!r}")
        return v

    @property
    def primary(self) -> Publication | None:
        return next((p for p in self.publications if p.role == "primary"), None) or \
            (self.publications[0] if self.publications else None)
