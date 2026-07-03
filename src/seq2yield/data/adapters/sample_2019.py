"""Sample et al. 2019 adapter (K6) — human 5'UTR MPRA -> mean ribosome load.

Data: GEO GSE114002 + github.com/pjsample/human_5utr_modeling (place CSV(s) under
data/extracted/sample_2019/). Defaults (docs/ONBOARDING.md §9): author read-count filter (keep the
best-measured ~280k), raw MRL target, provided 260k/20k split. Column names are detected
defensively because the GEO/GitHub CSVs vary slightly.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..cleaning import SEQ_COL, TARGET_COL, VALID_BASES

ROOT = Path(__file__).resolve().parents[4]
_N_KEEP = 280_000        # author-style read-count filter (best-measured)
_N_TEST = 20_000         # held-out test (260k train / 20k test)


def _pick(cols, *candidates):
    low = {c.lower(): c for c in cols}
    for c in candidates:
        if c in low:
            return low[c]
    return None


def load(spec):
    local = ROOT / spec.source.get("local", "data/extracted/sample_2019")
    csvs = sorted(local.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(
            f"no Sample-2019 CSVs under {local}. Download GEO {spec.source.get('geo')} "
            f"({spec.source.get('repo')}) and place the eGFP library CSV(s) there "
            "(see docs/ONBOARDING.md).")
    return pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)


def clean(df, spec) -> pd.DataFrame:
    seq_c = _pick(df.columns, "utr", "sequence", "seq")
    rl_c = _pick(df.columns, "rl", "mean_ribosome_load", "mrl", spec.target_col.lower())
    reads_c = _pick(df.columns, "total_reads", "total", "reads", "count")
    if seq_c is None or rl_c is None:
        raise ValueError(f"could not find sequence/target columns in {list(df.columns)}")

    out = pd.DataFrame({SEQ_COL: df[seq_c].astype(str).str.strip().str.upper(),
                        TARGET_COL: pd.to_numeric(df[rl_c], errors="coerce")})
    if reads_c is not None:
        out["_reads"] = pd.to_numeric(df[reads_c], errors="coerce").fillna(0)

    valid = ((out[SEQ_COL].str.len() == spec.seq_len)
             & out[SEQ_COL].apply(lambda s: set(s) <= VALID_BASES)
             & out[TARGET_COL].notna())
    out = out[valid].reset_index(drop=True)

    if "_reads" in out.columns:                       # author read-count filter: keep best-measured
        out = out.sort_values("_reads", ascending=False).head(_N_KEEP).reset_index(drop=True)
        out = out.drop(columns=["_reads"])

    # provided split: seeded random 20k held-out, rest train (matches 260k/20k)
    rng = np.random.default_rng(1)
    n_test = min(_N_TEST, max(1, len(out) // 10))
    test_idx = set(rng.choice(out.index.to_numpy(), size=n_test, replace=False).tolist())
    out["split"] = ["test" if i in test_idx else "train" for i in out.index]
    return out[[SEQ_COL, TARGET_COL, "split"]]
