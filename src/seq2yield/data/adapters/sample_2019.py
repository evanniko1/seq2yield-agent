"""Sample et al. 2019 adapter (K6) — human 5'UTR MPRA -> mean ribosome load.
Data: GEO GSE114002 + github.com/pjsample/human_5utr_modeling. Author read-count filter (~280k)
+ provided 260k/20k split (docs/ONBOARDING.md §9). Shares the Seelig 5'UTR cleaner.
"""
from __future__ import annotations

from . import _seelig


def load(spec):
    return _seelig.load_csvs(spec)


def clean(df, spec):
    return _seelig.clean_utr_mpra(df, spec, n_keep=280_000, n_test=20_000)
