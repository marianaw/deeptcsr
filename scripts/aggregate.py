"""Aggregate Hydra-run outputs into a tidy DataFrame.

Walks ``outputs/<algorithm>/<dataset>/<backbone>/lambda_*/target_lr_*/landmark_*/seed_*/``
and pairs each ``results_val.json`` with ``results_test.json``. Per (algorithm,
dataset, seed), the hyper-parameter combo with the best validation C-Index is
selected. This produces one row per seed that can then be reduced to mean/std
or fed into a paired test.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


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


def best_val_per_seed(df: pd.DataFrame, metric: str = "val_ci",
                       maximize: bool = True) -> pd.DataFrame:
    """For each (algorithm, dataset, seed), keep the row with best val metric."""
    grp = df.groupby(["algorithm", "dataset", "seed"], as_index=False)
    return grp.apply(lambda g: g.loc[g[metric].idxmax() if maximize
                                    else g[metric].idxmin()]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs", type=Path)
    ap.add_argument("--out", default="outputs/results.csv", type=Path)
    args = ap.parse_args()
    df = collect(args.root)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()
