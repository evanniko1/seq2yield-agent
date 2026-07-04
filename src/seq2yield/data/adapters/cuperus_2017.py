"""Cuperus et al. 2017 adapter (K6) — yeast 5'UTR MPRA -> ribosome load (growth selection).
Data: GEO GSE104252 (~500k random 50nt 5'UTRs). Same Seelig format as Sample 2019; no author
provided split -> downstream target-stratified holdout.
"""
from __future__ import annotations

from . import _seelig


def load(spec):
    return _seelig.load_csvs(spec)


def clean(df, spec):
    return _seelig.clean_utr_mpra(df, spec, n_keep=500_000, n_test=None)   # -> stratified holdout
