"""Figure: small-data learning curves (CI and IBS vs number of sequences).

Reads ``results/small_data/summary.csv`` (written by small_data_table.py;
hyper-parameters selected on validation separately per metric) and renders
a 2x3 grid: rows = C-index / IBS, columns = PBC2 / AIDS / SmallRW, one
line with error bars per algorithm. Matches the paper's Figure-3 style.

Usage: uv run python scripts/plot_small_data.py [--errorbar sem|std]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DATASETS = [("pbc2", "PBC2"), ("aids", "AIDS"), ("rw", "SmallRW")]
METRICS = [("test_ci", "Concordance Index"), ("test_bs", "Integrated Brier Score")]
# color follows the paper's scheme; distinct markers keep the red/green
# pair separable for colorblind readers
STYLE = {
    "cox": ("SA Landmarking", "#1F77B4", "o"),
    "tcsr_fitted": ("TCSR (fitted)", "#2CA02C", "s"),
    "inc_tcsr": ("Inc-TCSR", "#D62728", "^"),
    "d_tcsr": ("D-TCSR", "#9467BD", "D"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="results/small_data/summary.csv",
                    type=Path)
    ap.add_argument("--out", default="results/small_data/small_data_curves",
                    type=Path, help="output path stem (.pdf and .png)")
    ap.add_argument("--errorbar", default="sem", choices=["sem", "std"])
    args = ap.parse_args()

    df = pd.read_csv(args.summary)
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5), sharex=True)

    for i, (metric, ylabel) in enumerate(METRICS):
        for j, (ds, title) in enumerate(DATASETS):
            ax = axes[i, j]
            for algo, (label, color, marker) in STYLE.items():
                g = df[(df.dataset == ds) & (df.algorithm == algo)
                       & (df.metric == metric)].sort_values("n_train")
                if g.empty:
                    continue
                ax.errorbar(g.n_train, g["mean"], yerr=g[args.errorbar],
                            label=label, color=color, marker=marker,
                            ms=5, ls="--", lw=1.4, capsize=3)
            ax.set_xticks(sorted(df.n_train.unique()))
            ax.grid(alpha=0.4)
            ax.set_axisbelow(True)
            if i == 0:
                ax.set_title(title)
            if i == 1:
                ax.set_xlabel("Nb. of sequences")
            if j == 0:
                ax.set_ylabel(ylabel)

    axes[0, 0].legend(loc="lower right", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        path = args.out.with_suffix(f".{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
