"""Secondary yeast benchmark (Vaishnav et al.): pooled YFP prediction from 80 nt promoters.

Yeast has ~20 sequences per gene (199 genes) — too few for per-gene models — so models are
trained POOLED on a held-out set of whole genes, with a FAMILY-level paired bootstrap. Produces a
yeast baseline registry + report, and a ranking-transfer note vs the E. coli registry.

The protocol is explicit because it changed. `--protocol grouped` (the default, correct one) holds
whole native genes out and resamples genes: each gene is a mutational series, so the original
per-gene *stratified* holdout put a one-base sibling of every held-out sequence into training and
then counted its ~20 members as 20 independent observations (audit SA-1/ST-7, DECISIONS #63).
`--protocol legacy` reproduces that original run verbatim, for the record — its numbers are
superseded; see docs/RETRACTIONS.md.

Usage: python scripts/build_yeast.py [--protocol grouped|legacy]
"""
from __future__ import annotations

import argparse
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
    """LEGACY (superseded): 10% out of *each* gene, so every held-out sequence has a sibling in
    train. Kept only so `--protocol legacy` can reproduce the original table."""
    rng = np.random.default_rng(seed)
    test_idx = []
    for _, grp in df.groupby(SERIES_COL):
        k = max(1, int(round(frac * len(grp))))
        test_idx += list(rng.choice(grp.index.to_numpy(), size=min(k, len(grp)), replace=False))
    test = df.loc[test_idx]
    train = df.drop(index=test_idx)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def grouped_holdout(df: pd.DataFrame, frac: float = 0.1, seed: int = SEED):
    """Hold WHOLE genes out, via the dataset spec's own family declaration (family_col:
    mut_series), so no held-out sequence has a relative in training."""
    from seq2yield.data import datasets, families
    from seq2yield.experiments import pooled_runner
    labels, how = families.family_labels(df, datasets.spec("yeast"), seq_col=SEQ_COL)
    if labels is None:
        raise SystemExit("yeast spec no longer declares design families — refusing to split blind")
    train, test = pooled_runner.grouped_holdout(df, labels, frac=frac, seed=seed)
    return train, test, how


def ecoli_ranking() -> list[str] | None:
    csv = ROOT / "experiments/runs/2026-06-04-full56/data_size_curve.csv"
    if not csv.exists():
        return None
    d = pd.read_csv(csv)
    big = d["train_size"].max()
    sub = d[d["train_size"] == big].sort_values("r2_mean", ascending=False)
    return sub["model"].tolist()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", choices=["grouped", "legacy"], default="grouped",
                    help="grouped: whole genes held out, family-unit bootstrap (correct). "
                         "legacy: the superseded per-gene stratified holdout + sequence unit.")
    args = ap.parse_args()
    if not YEAST_CSV.exists():
        print(f"ERROR: {YEAST_CSV} missing (run the audit/extract first)", file=sys.stderr)
        return 2
    df = clean_yeast(pd.read_csv(YEAST_CSV))
    print(f"[yeast] {len(df)} promoters, {df[SERIES_COL].nunique()} genes, "
          f"{YEAST_SEQ_LEN} nt, target range [{df[TARGET_COL].min():.1f}, {df[TARGET_COL].max():.1f}]")

    if args.protocol == "grouped":
        train, test, how = grouped_holdout(df)
        groups = test[SERIES_COL].to_numpy()
        unit, split_name = "family", f"grouped_holdout ({how})"
    else:
        train, test = stratified_holdout(df)
        groups, unit, split_name = None, "sequence", "stratified_holdout (SUPERSEDED)"
    n_units = len(set(groups)) if groups is not None else len(test)
    print(f"[yeast] protocol={args.protocol}: split={split_name}, bootstrap unit={unit} "
          f"({n_units} units)")
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
        ci = bootstrap_r2_ci(y_test, p, groups=groups, seed=SEED)
        results[m] = {"r2": round(ci["r2"], 4), "r2_ci": [round(x, 4) for x in ci["ci"]]}
        print(f"  {m:>4}: R2={ci['r2']:.4f}  95% CI {[round(x,3) for x in ci['ci']]}")

    ranking = sorted(results, key=lambda m: -results[m]["r2"])
    best = ranking[0]
    # is the best model significantly better than the runner-up?
    runner = ranking[1]
    pb = paired_bootstrap_r2(y_test, preds[best], preds[runner], groups=groups, seed=SEED)
    print(f"[yeast] best={best} vs {runner}: ΔR2={pb['mean_delta']:.4f} "
          f"CI {[round(x,3) for x in pb['ci']]} excludes0={pb['excludes_zero']}")

    eco = ecoli_ranking()
    transfer = None
    if eco:
        transfer = {"ecoli_ranking": eco, "yeast_ranking": ranking,
                    "top_model_agrees": eco[0] == best}
        print(f"[yeast] ranking transfer — E. coli {eco} vs yeast {ranking} "
              f"(top agrees: {eco[0] == best})")

    # The protocol decides where the artifacts land, and NEITHER writes to the original
    # `yeast-baseline/` folder: that is frozen June-2026 history, produced by an older version of
    # the model code, and re-running `legacy` today does not reproduce it (see RETRACTIONS.md R-2 —
    # rf and cnn swap places from model drift alone, audit ML-02). Overwriting it would destroy the
    # only record of what the retracted claim was actually based on.
    suffix = "-legacy-rerun" if args.protocol == "legacy" else "-grouped"
    out_dir = ROOT / f"experiments/runs/yeast-baseline{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    registry_doc = {"dataset": "yeast", "seq_len": YEAST_SEQ_LEN, "n_promoters": int(len(df)),
                    "n_genes": int(df[SERIES_COL].nunique()), "n_test": int(len(test)),
                    "protocol": {"name": args.protocol, "split": split_name,
                                 "bootstrap_unit": unit, "n_bootstrap_units": int(n_units),
                                 "superseded": args.protocol == "legacy",
                                 "see": "docs/RETRACTIONS.md"},
                    "models": results, "ranking": ranking,
                    "best_vs_runnerup": {"best": best, "runner_up": runner, **pb},
                    "ranking_transfer": transfer, "environment": environment()}
    (out_dir / "yeast_registry.json").write_text(json.dumps(registry_doc, indent=2), encoding="utf-8")
    report = _report(registry_doc, suffix)
    print(f"[yeast] wrote {out_dir/'yeast_registry.json'} and {report}")
    return 0


def _report(doc: dict, suffix: str = "") -> Path:
    out = ROOT / f"reports/static/yeast_baseline{suffix.replace('-', '_')}_report.md"
    p = doc["protocol"]
    banner = ([f"> ⚠️ **Superseded protocol** — {p['split']}, bootstrap unit "
               f"`{p['bootstrap_unit']}`. These numbers are kept for the record only; see "
               "[docs/RETRACTIONS.md](../../docs/RETRACTIONS.md).", ""]
              if p["superseded"] else [])
    lines = [
        "# Yeast secondary benchmark — pooled YFP prediction (80 nt promoters)", "", *banner,
        f"{doc['n_promoters']} promoters · {doc['n_genes']} native genes · pooled training · "
        f"{doc['n_test']} held-out test sequences. Split: {p['split']}; paired bootstrap over "
        f"**{p['n_bootstrap_units']} {p['bootstrap_unit']}** units.", "",
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
    return out.relative_to(ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
