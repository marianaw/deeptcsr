"""Summarize the large-dataset benchmark: baseline vs Inc-TCSR vs D-TCSR.

Sources:
  - New Inc-TCSR runs (this repo, ``outputs/inc_tc_cox``, ``outputs/inc_tc_ddh``).
  - Legacy reference results (``--legacy-root``, default ../survan/_legacy_results):
      cox/SA/<ds>/...                       -> baseline Cox
      cox/DeepLambdaSA/<ds>/...             -> D-TCSR Cox family
      ddh/<ds>/seed_*/results*.json         -> baseline DDH
      ddh_tc/<ds>/seed_*/lambda*/target_lr*/results.json -> D-TCSR DDH family

Hyper-parameters (lambda for Inc-TCSR; lambda x tau for D-TCSR) are selected
on validation separately per metric: val CI (max) -> test CI, val IBS (min)
-> test IBS.

PROVENANCE: the whole Cox family (baseline, Inc-TCSR, D-TCSR) was recomputed
with current code, so its IBS is internally consistent and free of the
off-by-one fixed in commit 6b488da. The DDH family still comes from the
legacy runs, whose IBS already used surv[:, 1:] and so is comparable.

Writes results/large_data/{all_runs.csv,summary.csv} and prints tables.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from aggregate import collect

DATASETS = ["nasa", "scania", "churn_lastfm_months", "big_rw"]
SELECTIONS = {"test_ci": ("val_ci", True), "test_bs": ("val_bs", False)}
LABELS = {
    "cox": "Cox (baseline)", "inc_tc_cox": "Inc-TCSR Cox", "tc_cox": "D-TCSR Cox",
    "ddh": "DDH (baseline)", "inc_tc_ddh": "Inc-TCSR DDH", "tc_ddh": "D-TCSR DDH",
}
FAMILIES = {"cox": ["cox", "inc_tc_cox", "tc_cox"],
            "ddh": ["ddh", "inc_tc_ddh", "tc_ddh"]}


def _num(path_part, key):
    m = re.search(rf"{key}_([\d.]+)", path_part)
    return float(m.group(1)) if m else np.nan


def load_legacy(root: Path) -> pd.DataFrame:
    rows = []
    # Cox family: results_val.json + results_test.json pairs.
    for algo_dir, algo in (("cox/SA", "cox"), ("cox/DeepLambdaSA", "tc_cox")):
        for f in (root / algo_dir).rglob("results_test.json"):
            fv = f.parent / "results_val.json"
            if not fv.exists():
                continue
            p = str(f.parent)
            test = json.load(open(f))
            val = json.load(open(fv))
            m = re.search(r"seed_(\d+)", p)
            ds = next((d for d in DATASETS if f"/{d}/" in p
                       or f"/{d}+" in p), None)
            if ds is None or m is None or "+100epochs" in p:
                continue
            rows.append({"algorithm": algo, "dataset": ds,
                         "seed": int(m.group(1)),
                         "lambda_": _num(p, "lambda"),
                         "target_lr": _num(p, "target_lr"),
                         "val_ci": val["ci"], "val_bs": val["bs"],
                         "test_ci": test["ci"], "test_bs": test["bs"],
                         "source": "legacy", "run_dir": p})
    # DDH family: single results.json per run.
    for algo_dir, algo in (("ddh", "ddh"), ("ddh_tc", "tc_ddh")):
        for f in (root / algo_dir).rglob("results*.json"):
            p = str(f.parent)
            if "_old" in p:
                continue
            d = json.load(open(f))
            m = re.search(r"seed_(\d+)", p)
            ds = next((x for x in DATASETS if f"/{x}/" in p), None)
            if ds is None or m is None:
                continue
            row = {"algorithm": algo, "dataset": ds, "seed": int(m.group(1)),
                   "lambda_": d.get("lambda_", _num(p, "lambda")),
                   "target_lr": d.get("target_lr", _num(p, "target_lr")),
                   "source": "legacy", "run_dir": p}
            if "test_ci" in d:
                row.update(val_ci=d.get("val_ci"), val_bs=d.get("val_ibs"),
                           test_ci=d["test_ci"], test_bs=d["test_ibs"])
            elif "ci" in d:  # baseline: test only; use test as val (no tuning)
                row.update(val_ci=d["ci"], val_bs=d.get("ibs"),
                           test_ci=d["ci"], test_bs=d.get("ibs"))
            else:
                continue
            rows.append(row)
    return pd.DataFrame(rows)


def load_new(root: Path) -> pd.DataFrame:
    df = collect(root)
    if df.empty:
        return df
    df = df[df.algorithm.isin(["inc_tc_cox", "inc_tc_ddh", "cox", "tc_cox"])
            & df.dataset.isin(DATASETS)]
    df["source"] = "new"
    return df


def select_per_metric(df, val_metric, maximize):
    grp = df.groupby(["algorithm", "dataset", "seed"], as_index=False)
    idx = grp.apply(lambda g: g[val_metric].idxmax() if maximize
                    else g[val_metric].idxmin()).iloc[:, -1]
    return df.loc[idx.astype(int)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs", type=Path)
    ap.add_argument("--legacy-root", type=Path,
                    default=Path("../survan/_legacy_results"))
    ap.add_argument("--out", default="results/large_data", type=Path)
    args = ap.parse_args()

    new = load_new(args.root)
    legacy = load_legacy(args.legacy_root)
    # New (current-code) runs supersede legacy ones for the same
    # (algorithm, dataset) — relevant for the Cox-family recompute.
    if not new.empty:
        have = set(map(tuple, new[["algorithm", "dataset"]].values))
        legacy = legacy[~legacy[["algorithm", "dataset"]]
                        .apply(tuple, axis=1).isin(have)]
    df = pd.concat([new, legacy], ignore_index=True)
    df = df.dropna(subset=["val_ci", "test_ci"])

    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "all_runs.csv", index=False)

    summaries = []
    for metric, (vm, mx) in SELECTIONS.items():
        sub = df.dropna(subset=[vm, metric])
        sel = select_per_metric(sub, vm, mx)
        for (algo, ds), g in sel.groupby(["algorithm", "dataset"]):
            summaries.append({"algorithm": algo, "dataset": ds,
                              "metric": metric, "n_seeds": len(g),
                              "mean": g[metric].mean(),
                              "sem": stats.sem(g[metric]) if len(g) > 1 else np.nan})
        for fam, algos in FAMILIES.items():
            print(f"\n=== {fam.upper()} family — "
                  f"{'Test C-index (sel. val CI)' if metric == 'test_ci' else 'Test IBS (sel. val IBS)'} ===")
            rows = []
            for algo in algos:
                row = {"algorithm": LABELS[algo]}
                for ds in DATASETS:
                    g = sel[(sel.algorithm == algo) & (sel.dataset == ds)]
                    row[ds] = (f"{g[metric].mean():.4f}±{stats.sem(g[metric]):.4f} ({len(g)})"
                               if len(g) > 1 else "—")
                rows.append(row)
            print(pd.DataFrame(rows).to_string(index=False))
            # paired test: Inc vs D-TCSR on common seeds
            inc, d = algos[1], algos[2]
            for ds in DATASETS:
                a = sel[(sel.algorithm == inc) & (sel.dataset == ds)].set_index("seed")[metric]
                b = sel[(sel.algorithm == d) & (sel.dataset == ds)].set_index("seed")[metric]
                common = a.index.intersection(b.index)
                if len(common) > 1:
                    t = stats.ttest_rel(b.loc[common], a.loc[common])
                    print(f"  {ds}: D-TCSR - Inc = {(b.loc[common] - a.loc[common]).mean():+.4f} "
                          f"(p={t.pvalue:.4f}, n={len(common)})")

    pd.DataFrame(summaries).to_csv(args.out / "summary.csv", index=False)
    print(f"\nWrote {args.out}/all_runs.csv and summary.csv")
    print("NOTE: Cox family = recomputed with current code (IBS internally "
          "consistent). DDH family = legacy runs; mimic/nasa are missing a "
          "seed there, so those cells average 10 seeds, not 11.")


if __name__ == "__main__":
    main()
