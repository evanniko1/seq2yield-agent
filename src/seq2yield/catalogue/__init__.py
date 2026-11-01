"""Dataset catalogue: every sequence-to-expression dataset the project knows about — registered,
scheduled, candidate, watched or rejected — with publications, accessions, fit verdicts and
verification state. Truth: data/catalogue/datasets.jsonl; projected into the project SQLite store."""
from .core import (CATALOGUE_PATH, AddResult, add, attach_accession, attach_publication, find_match,
                   get, load, norm_accession, save, score, set_status, suggest_id, sync_registry)
from .model import STATUSES, VERDICTS, Accession, CatalogueEntry, Publication

__all__ = ["CATALOGUE_PATH", "AddResult", "add", "attach_accession", "attach_publication",
           "find_match", "get", "load", "norm_accession", "save", "score", "set_status",
           "suggest_id", "sync_registry",
           "STATUSES", "VERDICTS", "Accession", "CatalogueEntry", "Publication"]
