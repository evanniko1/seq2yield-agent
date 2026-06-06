"""Secondary yeast benchmark (Vaishnav et al.): pooled YFP prediction from 80 nt promoters.

Yeast has ~20 sequences per gene (199 genes) — too few for per-gene models — so models are
trained POOLED, evaluated on a per-gene-stratified held-out set, with a SEQUENCE-LEVEL paired
bootstrap (the pooled-dataset analog of the E. coli per-series test). Produces a yeast baseline
registry + report, and a ranking-transfer note vs the E. coli registry.

Usage: python scripts/build_yeast.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from seq2yield.data.cleaning import SEQ_COL, SERIES_COL, TARGET_COL, YEAST_SEQ_LEN, clean_yeast  # noqa: E402
from seq2yield.models import registry  # noqa: E402
from seq2yield.statistics.bootstrap import bootstrap_r2_ci, paired_bootstrap_r2  # noqa: E402
from seq2yield.training.reproducibility import environment, set_seed  # noqa: E402
from seq2yield.training.train import features_for  # noqa: E402

YEAST_CSV = ROOT / "data/extracted/seq2yield/to_import/yeast_data.csv"
MODELS = ["rf", "mlp", "cnn"]
SEED = 1


def stratified_holdout(df: pd.DataFrame, frac: float = 0.1, seed: int = SEED):
    rng = np.random.default_rng(seed)
    test_idx = []
    for _, grp in df.groupby(SERIES_COL):
        k = max(1, int(round(frac * len(grp))))
        test_idx += list(rng.choice(grp.index.to_numpy(), size=min(k, len(grp)), replace=False))
    test = df.loc[test_idx]
    train = df.drop(index=test_idx)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def ecoli_ranking() -> list[str] | None:
    csv = ROOT / "experiments/runs/2026-06-04-full56/data_size_curve.csv"
    if not csv.exists():
        return None
    d = pd.read_csv(csv)
    big = d["train_size"].max()
    sub = d[d["train_size"] == big].sort_values("r2_mean", ascending=False)
    return sub["model"].tolist()


def main() -> int:
    if not YEAST_CSV.exists():
        print(f"ERROR: {YEAST_CSV} missing (run the audit/extract first)", file=sys.stderr)
        return 2
    df = clean_yeast(pd.read_csv(YEAST_CSV))
    print(f"[yeast] {len(df)} promoters, {df[SERIES_COL].nunique()} genes, "
          f"{YEAST_SEQ_LEN} nt, target range [{df[TARGET_COL].min():.1f}, {df[TARGET_COL].max():.1f}]")
    train, test = stratified_holdout(df)
    print(f"[yeast] pooled train={len(train)}  held-out test={len(test)}")
    y_test = test[TARGET_COL].to_numpy()

    results, preds = {}, {}
    for m in MODELS:
        set_seed(SEED)
        Xtr = features_for(m, train, "one_hot", YEAST_SEQ_LEN)
        Xte = features_for(m, test, "one_hot", YEAST_SEQ_LEN)
        model = registry.make(m, seed=SEED)
        model.fit(Xtr, train[TARGET_COL].to_numpy())
        p = model.predict(Xte)
        preds[m] = p
        ci = bootstrap_r2_ci(y_test, p, seed=SEED)
        results[m] = {"r2": round(ci["r2"], 4), "r2_ci": [round(x, 4) for x in ci["ci"]]}
        print(f"  {m:>4}: R2={ci['r2']:.4f}  95% CI {[round(x,3) for x in ci['ci']]}")

    ranking = sorted(results, key=lambda m: -results[m]["r2"])
    best = ranking[0]
    # is the best model significantly better than the runner-up? (sequence-level paired bootstrap)
    runner = ranking[1]
    pb = paired_bootstrap_r2(y_test, preds[best], preds[runner], seed=SEED)
    print(f"[yeast] best={best} vs {runner}: ΔR2={pb['mean_delta']:.4f} "
          f"CI {[round(x,3) for x in pb['ci']]} excludes0={pb['excludes_zero']}")

    eco = ecoli_ranking()
    transfer = None
    if eco:
        transfer = {"ecoli_ranking": eco, "yeast_ranking": ranking,
                    "top_model_agrees": eco[0] == best}
        print(f"[yeast] ranking transfer — E. coli {eco} vs yeast {ranking} "
              f"(top agrees: {eco[0] == best})")

    out_dir = ROOT / "experiments/runs/yeast-baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    registry_doc = {"dataset": "yeast", "seq_len": YEAST_SEQ_LEN, "n_promoters": int(len(df)),
                    "n_genes": int(df[SERIES_COL].nunique()), "n_test": int(len(test)),
                    "models": results, "ranking": ranking,
                    "best_vs_runnerup": {"best": best, "runner_up": runner, **pb},
                    "ranking_transfer": transfer, "environment": environment()}
    (out_dir / "yeast_registry.json").write_text(json.dumps(registry_doc, indent=2), encoding="utf-8")
    _report(registry_doc)
    print(f"[yeast] wrote {out_dir/'yeast_registry.json'} and reports/static/yeast_baseline_report.md")
    return 0


def _report(doc: dict) -> None:
    out = ROOT / "reports/static/yeast_baseline_report.md"
    lines = [
        "# Yeast secondary benchmark — pooled YFP prediction (80 nt promoters)", "",
        f"{doc['n_promoters']} promoters · {doc['n_genes']} native genes · pooled training · "
        f"{doc['n_test']} held-out test sequences (per-gene stratified). Sequence-level "
        "bootstrap (pooled analog of the E. coli per-series test).", "",
        "## Per-model R² (held-out)", "",
        "| model | R² | 95% CI |", "| --- | --- | --- |",
    ]
    for m, d in sorted(doc["models"].items(), key=lambda kv: -kv[1]["r2"]):
        lines.append(f"| {m} | {d['r2']} | {d['r2_ci']} |")
    b = doc["best_vs_runnerup"]
    lines += ["", f"**Best:** {b['best']} vs {b['runner_up']} — ΔR²={round(b['mean_delta'],4)}, "
              f"95% CI {[round(x,3) for x in b['ci']]}, excludes 0: {b['excludes_zero']}", ""]
    if doc.get("ranking_transfer"):
        t = doc["ranking_transfer"]
        lines += ["## Cross-organism ranking transfer",
                  f"- E. coli ranking (R²@2000): {t['ecoli_ranking']}",
                  f"- Yeast ranking: {t['yeast_ranking']}",
                  f"- **Top model agrees across organisms: {t['top_model_agrees']}**",
                  "- (Direct weight transfer is not possible — 96 nt vs 80 nt one-hot dims "
                  "differ; this compares the model *ranking*, a transfer-of-conclusions question.)"]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
