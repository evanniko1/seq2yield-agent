"""Reporting: data-size curve plot + markdown tables for the reproduction report."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def data_size_curve_png(agg: pd.DataFrame, out_path: str | Path, title: str) -> Path | None:
    """agg columns: model, train_size, r2_mean, r2_std. Returns path or None if plotting fails."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    fig, ax = plt.subplots(figsize=(7, 5))
    for model, g in agg.groupby("model"):
        g = g.sort_values("train_size")
        ax.errorbar(g["train_size"], g["r2_mean"], yerr=g["r2_std"], marker="o",
                    capsize=3, label=model)
    ax.set_xlabel("training set size (per series)")
    ax.set_ylabel(r"mean held-out $R^2$ (across series)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def df_to_markdown(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:  # tabulate missing -> simple pipe table
        cols = list(df.columns)
        lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
        for _, row in df.iterrows():
            lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
        return "\n".join(lines)
