"""C8 — joint / cross-dataset training via pluggable LENGTH-RECONCILIATION.

Our datasets are 50/80/96/110/270 nt, so one model cannot consume them jointly in raw one-hot. But
length is a *representation* problem with several standard fixes — not only foundation-model
embeddings. This module pools datasets and does train-on-A→test-on-B in a common FIXED-WIDTH space,
with the reconciliation strategy selectable (all four are testable; `kmer` is the default):

  • kmer          — k-mer spectrum (4^k dims): length-invariant by construction. DEFAULT.
  • pad           — one-hot padded/truncated to a common max length + a pad-indicator channel.
  • adaptive_pool — per-bin base composition (4·B dims): the length-agnostic global-pool analog
                    for non-deep models (bin the sequence into B segments, take base fractions).
  • embed         — mean-pooled frozen foundation-model embedding (K2a; needs the cache).

Because each dataset is a DIFFERENT assay, targets are z-scored PER DATASET before pooling, and the
cross-assay metric is Spearman (rank, scale-free) alongside R² on the z-target — so "does a model
trained on A rank B correctly" is answered honestly.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ..data import datasets
from ..data.cleaning import SEQ_COL, TARGET_COL
from ..models import registry as model_registry
from ..training import metrics as M
from . import pooled_runner

ROOT = Path(__file__).resolve().parents[3]
STRATEGIES = ("kmer", "pad", "adaptive_pool", "embed")
_BASES = {"A": 0, "C": 1, "G": 2, "T": 3}


@dataclass
class JointResult:
    train_datasets: list[str]
    test_dataset: str
    model: str
    strategy: str
    feature_dim: int
    n_train: int
    n_test: int
    spearman: float                 # scale-free cross-assay metric (primary)
    r2_z: float                     # R² on the per-dataset z-scored target
    metric_primary: str = "spearman"

    def as_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------ reconciliation backends
def _kmer(seqs, k: int = 4) -> np.ndarray:
    dim = 4 ** k
    out = np.zeros((len(seqs), dim), dtype=np.float32)
    for i, s in enumerate(seqs):
        s = s.upper()
        for j in range(len(s) - k + 1):
            idx, ok = 0, True
            for b in s[j:j + k]:
                v = _BASES.get(b)
                if v is None:
                    ok = False
                    break
                idx = idx * 4 + v
            if ok:
                out[i, idx] += 1.0
        n = out[i].sum()
        if n:
            out[i] /= n                                   # frequency, length-normalized
    return out


def _pad(seqs, max_len: int) -> np.ndarray:
    out = np.zeros((len(seqs), 5 * max_len), dtype=np.float32)  # 4 bases + 1 pad channel
    for i, s in enumerate(seqs):
        s = s.upper()[:max_len]
        for p in range(max_len):
            base = _BASES.get(s[p]) if p < len(s) else None
            out[i, p * 5 + (base if base is not None else 4)] = 1.0
    return out


def _binned(seqs, n_bins: int = 8) -> np.ndarray:
    out = np.zeros((len(seqs), 4 * n_bins), dtype=np.float32)
    for i, s in enumerate(seqs):
        s = s.upper()
        L = max(1, len(s))
        for p, ch in enumerate(s):
            v = _BASES.get(ch)
            if v is None:
                continue
            b = min(n_bins - 1, p * n_bins // L)
            out[i, b * 4 + v] += 1.0
        for b in range(n_bins):
            seg = out[i, b * 4:(b + 1) * 4]
            tot = seg.sum()
            if tot:
                out[i, b * 4:(b + 1) * 4] = seg / tot     # per-bin base fractions
    return out


def reconcile(seqs, strategy: str, *, dataset: str | None = None, k: int = 4, max_len: int = 300,
              n_bins: int = 8, embed_model: str | None = None) -> np.ndarray:
    """Fixed-width feature matrix for `seqs` under `strategy` (same width for every dataset)."""
    if strategy == "kmer":
        return _kmer(seqs, k)
    if strategy == "pad":
        return _pad(seqs, max_len)
    if strategy == "adaptive_pool":
        return _binned(seqs, n_bins)
    if strategy == "embed":
        if not (dataset and embed_model):
            raise ValueError("embed strategy needs dataset + embed_model")
        from ..features.embeddings import embedding_features
        return np.asarray(embedding_features(embed_model, list(seqs), dataset), dtype=np.float32)
    raise ValueError(f"unknown strategy '{strategy}' (use {STRATEGIES})")


# ------------------------------------------------------------------ frames + z-target
def _dataset_frames(dataset: str, seed: int):
    ds = datasets.spec(dataset)
    if ds.structure == "per_series":
        from ..data.loaders import load_split_csv
        from ..data.splits import load_manifest
        man = load_manifest(ROOT / "data/splits")
        it = next(iter(man["iterations"]))
        return (load_split_csv(man["iterations"][it]["working_set"]["path"]),
                load_split_csv(man["iterations"][it]["heldout_set"]["path"]))
    return pooled_runner.holdout(SimpleNamespace(dataset=dataset, seed=seed))


def _z(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    s = y.std() or 1.0
    return (y - y.mean()) / s


# ------------------------------------------------------------------ joint train / eval
def run_joint(train_datasets: list[str], test_dataset: str, *, model: str = "rf",
              strategy: str = "kmer", train_size_per: int = 2000, k: int = 4, n_bins: int = 8,
              embed_model: str | None = None, seed: int = 0, record: bool = False) -> JointResult:
    """Pool `train_datasets` in the reconciled space, train `model`, and predict `test_dataset`'s
    held-out set (train-on-A→test-on-B across assays). Flat models only (the reconciled X is flat)."""
    if model_registry.feature_kind(model) != "flat":
        raise ValueError(f"joint currently supports flat models on reconciled features, not '{model}'")
    involved = list(dict.fromkeys([*train_datasets, test_dataset]))
    max_len = max(datasets.seq_len(d) for d in involved)  # common pad width across the run
    rec = lambda ds, seqs: reconcile(seqs, strategy, dataset=ds, k=k, max_len=max_len,
                                     n_bins=n_bins, embed_model=embed_model)

    Xtr_parts, ytr_parts = [], []
    for ds in train_datasets:
        train, _test = _dataset_frames(ds, seed)
        train = pooled_runner.subsample(train, train_size_per, "expression_stratified", seed)
        Xtr_parts.append(rec(ds, train[SEQ_COL].tolist()))
        ytr_parts.append(_z(train[TARGET_COL].to_numpy()))       # z-score per dataset before pooling
    Xtr = np.vstack(Xtr_parts)
    ytr = np.concatenate(ytr_parts)

    _tr, test = _dataset_frames(test_dataset, seed)
    Xte = rec(test_dataset, test[SEQ_COL].tolist())
    yte_z = _z(test[TARGET_COL].to_numpy())

    if model in ("ridge", "svr", "mlp"):                          # scale-sensitive flat models
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler().fit(Xtr)
        Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)

    est = model_registry.make(model, seed=seed)
    est.fit(Xtr, ytr)
    pred = est.predict(Xte)

    res = JointResult(
        train_datasets=list(train_datasets), test_dataset=test_dataset, model=model,
        strategy=strategy, feature_dim=int(Xtr.shape[1]), n_train=int(len(ytr)),
        n_test=int(len(yte_z)), spearman=round(float(M.spearman(yte_z, pred)), 4),
        r2_z=round(float(M.r2(yte_z, pred)), 4))
    if record:
        record_joint(res)
    return res


def compare_strategies(train_datasets: list[str], test_dataset: str, *, model: str = "rf",
                       strategies=("kmer", "adaptive_pool"), **kw) -> list[JointResult]:
    """Run several reconciliation strategies on the same joint task (so they can be compared)."""
    out = []
    for s in strategies:
        try:
            out.append(run_joint(train_datasets, test_dataset, model=model, strategy=s, **kw))
        except Exception as e:                                    # e.g. embed cache absent
            out.append(JointResult(list(train_datasets), test_dataset, model, s, 0, 0, 0,
                                   float("nan"), float("nan"), metric_primary=f"error:{str(e)[:40]}"))
    return out


def record_joint(res: JointResult, out_dir: str | Path | None = None) -> Path:
    import json
    d = Path(out_dir or ROOT / "reports" / "joint")
    d.mkdir(parents=True, exist_ok=True)
    src = "+".join(res.train_datasets)
    p = d / f"{src}-to-{res.test_dataset}-{res.model}-{res.strategy}-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}.json"
    p.write_text(json.dumps(res.as_dict(), indent=2), encoding="utf-8")
    return p
