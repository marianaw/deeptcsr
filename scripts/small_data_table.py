"""Summarize the small-data learning-curve benchmark.

Layout: ``outputs_small/ntrain_<n>/<algorithm>/<dataset>/...``. Hyper-
parameters (tau for D-TCSR, l2 for fitted TCSR — both live in the
``target_lr`` path slot) are selected on validation SEPARATELY per metric,
ONCE per (algorithm, dataset, n_train) on the mean over seeds: best val
C-index for the reported test C-index, best (lowest) val IBS for the
reported test IBS. That config's test metrics are averaged across seeds.

Writes tidy CSVs under ``results/small_data/`` for later querying:
  - all_runs.csv           every finished run, one row per hyperparameter combo
  - selected_val_ci.csv    per-seed rows of the config chosen on mean val CI
  - selected_val_ibs.csv   per-seed rows of the config chosen on mean val IBS
  - summary.csv            mean/sem per (dataset, algorithm, n_train, metric)

Usage: uv run python scripts/small_data_table.py [--root outputs_small]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from selection import pick_global


_PARTS = ("algorithm", "dataset", "backbone", "lambda_",
          "target_lr", "landmark", "seed")
_NUM_FIELDS = {"lambda_", "target_lr", "seed"}
_PATTERN = re.compile(r"(lambda|target_lr|landmark|seed)_(.+)")


def _parse_run_dir(p: Path):
    rel = p.relative_to(p.parents[6])
    if len(rel.parts) != len(_PARTS):
        return None
    algo, ds, bb = rel.parts[:3]
    out = {"algorithm": algo, "dataset": ds, "backbone": bb}
    for raw, key in zip(rel.parts[3:], _PARTS[3:]):
        m = _PATTERN.fullmatch(raw)
        if not m:
            return None
        k, v = m.groups()
        k = "lambda_" if k == "lambda" else k
        out[k] = float(v) if k in _NUM_FIELDS else v
    out["seed"] = int(out["seed"])
    return out


def _read_split(run: Path, split: str):
    f = run / f"results_{split}.json"
    if not f.exists():
        return None
    with open(f) as fh:
        d = json.load(fh)
    return {f"{split}_{k}": v for k, v in d.items() if k in ("ci", "bs")}


def collect(root: Path) -> pd.DataFrame:
    """Return one row per finished run (val+test pair)."""
    rows = []
    for run in root.glob("*/*/*/lambda_*/target_lr_*/landmark_*/seed_*"):
        meta = _parse_run_dir(run)
        if meta is None:
            continue
        val = _read_split(run, "val")
        test = _read_split(run, "test")
        if val is None or test is None:
            continue
        rows.append({**meta, **val, **test, "run_dir": str(run)})
    return pd.DataFrame(rows)


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


def select(df: pd.DataFrame, val_metric: str, test_metric: str,
           maximize: bool) -> pd.DataFrame:
    """One config per (algorithm, dataset, n_train), chosen on MEAN validation
    across seeds; returns that config's rows for every seed (same rule as the
    large benchmark, scripts/selection.py). Per-seed selection would let each
    split pick its own winner, i.e. tune the seed."""
    return pick_global(df, ["algorithm", "dataset", "n_train"],
                       val_metric, test_metric, maximize)


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
    selections = {m: select(df, vm, m, mx)
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
