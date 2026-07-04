"""One-off: the three 'optional' Sample-2019 deep-dives, run sequentially (GPU-friendly).
  1. Data-efficiency curve for CNN on human 5'UTR (toward the paper's ~0.9 R2).
  2. Foundation-model embedding (nt-50m) vs one_hot for RF on 5'UTR.
  3. Cross-dataset transfer: a properly-powered E. coli CNN>RF finding replicated on human 5'UTR.
Prints a summary; not part of the test suite.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np  # noqa: E402

from orchestration import execution_harness as H  # noqa: E402
from seq2yield.experiments.run_spec import AcceptancePolicy, RunSpec  # noqa: E402


def _rm(*ids):
    for r in ids:
        shutil.rmtree(ROOT / "experiments/runs" / r, ignore_errors=True)


def data_efficiency():
    print("\n[1] CNN data-efficiency on human 5'UTR (cnn vs rf) ...", flush=True)
    spec = RunSpec(run_id="opt-sample2019-cnn-curve", dataset="sample_2019",
                   intervention_type="model_architecture", model_family="cnn",
                   train_sizes=[2000, 20000, 100000], iterations=[1], seed=1, maturity_tier="tier_1",
                   acceptance_policy=AcceptancePolicy(track="performance", baseline_run_id="s2019",
                                                      baseline_model="rf", comparison_train_size=100000))
    c = H.run(spec, changed_files=[], run_tests=False)["comparison"]
    for p in c.get("per_size", []):
        print(f"    N={p['train_size']:>6}: cnn R2={p['candidate_mean']:.3f}  rf R2={p['baseline_mean']:.3f}",
              flush=True)
    _rm("opt-sample2019-cnn-curve")


def embedding_vs_onehot():
    print("\n[2] Foundation-model embedding (nt-50m) vs one_hot for RF on 5'UTR ...", flush=True)
    from seq2yield.data import adapters
    from seq2yield.embeddings import cache, extract
    from seq2yield.training.train import train_evaluate

    df = adapters.frame_for("sample_2019")
    train = df[df["split"] == "train"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)
    # bounded subset so extraction is quick; cover exactly the sequences we score
    tr = train.sample(n=4000, random_state=1).reset_index(drop=True)
    te = test.sample(n=2000, random_state=1).reset_index(drop=True)
    seqs = list(dict.fromkeys(tr["Sequence"].tolist() + te["Sequence"].tolist()))
    print(f"    extracting nt-50m for {len(seqs)} sequences ...", flush=True)
    vecs = extract.embed("nt-50m", seqs, batch_size=32)
    cache.write("nt-50m", "sample_2019", seqs, vecs)
    for fs in ("one_hot", "embed:nt-50m"):
        r = train_evaluate("rf", tr, te, feature_set=fs, length=50, seed=1, dataset="sample_2019")
        print(f"    rf [{fs:14s}] R2={r['r2']:.4f}", flush=True)


def transfer_powered():
    print("\n[3] Powered E. coli cnn>rf finding -> replicate on human 5'UTR ...", flush=True)
    src = RunSpec(run_id="opt-xfer-src", dataset="ecoli", intervention_type="model_architecture",
                  model_family="cnn", train_sizes=[2000], iterations=[1, 2], n_series=30, seed=1,
                  maturity_tier="tier_1",
                  acceptance_policy=AcceptancePolicy(track="performance", baseline_run_id="2026-06-04-full56",
                                                     baseline_model="rf", comparison_train_size=2000))
    sc = H.run(src, changed_files=[], run_tests=False)["comparison"]
    print(f"    E. coli source: delta={sc['mean_delta']:.4f} excl0={sc['ci_excludes_zero']}", flush=True)
    tgt = RunSpec(run_id="opt-xfer-tgt", dataset="sample_2019", intervention_type="model_architecture",
                  model_family="cnn", transfer_of_run_id="opt-xfer-src", transfer_source_dataset="ecoli",
                  train_sizes=[2000], iterations=[1], seed=1, maturity_tier="tier_1",
                  acceptance_policy=AcceptancePolicy(track="performance", baseline_run_id="s2019",
                                                     baseline_model="rf", comparison_train_size=2000))
    t = H.run(tgt, changed_files=[], run_tests=False)["comparison"]["transfer"]
    print(f"    TRANSFER VERDICT: {t['verdict'].upper()} — {t['reason']}", flush=True)
    _rm("opt-xfer-src", "opt-xfer-tgt")


def main() -> int:
    transfer_powered()        # cheapest first
    embedding_vs_onehot()
    data_efficiency()         # heaviest last
    print("\n[done] Sample-2019 optional deep-dives complete.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
