"""Summarize the small-data learning-curve benchmark.

Layout: ``outputs_small/ntrain_<n>/<algorithm>/<dataset>/...``. Hyper-
parameters (tau for D-TCSR, l2 for fitted TCSR — both live in the
``target_lr`` path slot) are selected on validation SEPARATELY per metric:
best val C-index for the reported test C-index, best (lowest) val IBS for
the reported test IBS. Test metrics are then averaged across seeds.

Writes tidy CSVs under ``results/small_data/`` for later querying:
  - all_runs.csv           every finished run, one row per hyperparameter combo
  - selected_val_ci.csv    per-seed rows after best-val-CI selection
  - selected_val_ibs.csv   per-seed rows after best-val-IBS selection
  - summary.csv            mean/sem per (dataset, algorithm, n_train, metric)

Usage: uv run python scripts/small_data_table.py [--root outputs_small]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from aggregate import best_val_per_seed, collect

ALGOS = ["cox", "tcsr_fitted", "inc_tcsr", "d_tcsr"]
LABELS = {"cox": "Baseline (landmark MLE)", "tcsr_fitted": "Fitted TCSR",
          "inc_tcsr": "Inc-TCSR (tau=1)", "d_tcsr": "D-TCSR (tau<1)"}
DATASETS = ["aids", "pbc2", "rw"]
# For CI select on val CI (max); for IBS select on val IBS (min).
SELECTIONS = {"test_ci": ("val_ci", True), "test_bs": ("val_bs", False)}


def load_all(root: Path) -> pd.DataFrame:
    frames = []
    for d in sorted(root.glob("ntrain_*"), key=lambda p: int(p.name.split("_")[1])):
        df = collect(d)
        if not df.empty:
            df["n_train"] = int(d.name.split("_")[1])
            frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    return df[df.algorithm.isin(ALGOS) & df.dataset.isin(DATASETS)]


def select(df: pd.DataFrame, val_metric: str, maximize: bool) -> pd.DataFrame:
    parts = []
    for n, g in df.groupby("n_train"):
        s = best_val_per_seed(g, metric=val_metric, maximize=maximize)
        s["n_train"] = n
        parts.append(s)
    return pd.concat(parts, ignore_index=True)


def summarize(selections: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for test_metric, sel in selections.items():
        for (ds, algo, n), g in sel.groupby(["dataset", "algorithm", "n_train"]):
            rows.append({
                "dataset": ds, "algorithm": algo, "n_train": n,
                "metric": test_metric, "n_seeds": len(g),
                "mean": g[test_metric].mean(),
                "sem": stats.sem(g[test_metric]),
                "std": g[test_metric].std(ddof=1),
            })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs_small", type=Path)
    ap.add_argument("--out", default="results/small_data", type=Path)
    args = ap.parse_args()

    df = load_all(args.root)
    selections = {m: select(df, vm, mx)
                  for m, (vm, mx) in SELECTIONS.items()}
    summary = summarize(selections)

    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "all_runs.csv", index=False)
    selections["test_ci"].to_csv(args.out / "selected_val_ci.csv", index=False)
    selections["test_bs"].to_csv(args.out / "selected_val_ibs.csv", index=False)
    summary.to_csv(args.out / "summary.csv", index=False)
    print(f"Wrote {len(df)} runs + selections + summary to {args.out}/\n")

    sizes = sorted(df.n_train.unique())
    for ds in DATASETS:
        print(f"\n=== {ds} ===")
        for metric, name in (("test_ci", "Test C-index (tau/l2 selected on val CI)"),
                             ("test_bs", "Test IBS (tau/l2 selected on val IBS)")):
            sel = selections[metric]
            sub = sel[sel.dataset == ds]
            rows = []
            for algo in ALGOS:
                row = {"algorithm": LABELS[algo]}
                for n in sizes:
                    g = sub[(sub.algorithm == algo) & (sub.n_train == n)]
                    row[f"n={n}"] = (f"{g[metric].mean():.3f}±{stats.sem(g[metric]):.3f}"
                                     if len(g) else "—")
                rows.append(row)
            print(f"\n{name}:")
            print(pd.DataFrame(rows).to_string(index=False))

        print("\nD-TCSR vs Inc-TCSR (paired, per-metric selection):")
        for n in sizes:
            out = [f"n={n:>3}"]
            for metric in ("test_ci", "test_bs"):
                sel = selections[metric]
                sub = sel[sel.dataset == ds]
                d = sub[(sub.algorithm == "d_tcsr") & (sub.n_train == n)].set_index("seed")
                r = sub[(sub.algorithm == "inc_tcsr") & (sub.n_train == n)].set_index("seed")
                common = d.index.intersection(r.index)
                if len(common) < 2:
                    continue
                diff = (d.loc[common, metric] - r.loc[common, metric]).mean()
                p = stats.ttest_rel(d.loc[common, metric],
                                    r.loc[common, metric]).pvalue
                out.append(f"{metric}: diff={diff:+.4f} p={p:.4f}")
            print("  " + "  ".join(out))


if __name__ == "__main__":
    main()
