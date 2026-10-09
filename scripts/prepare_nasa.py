"""Build data/NASA.h5 from the NASA C-MAPSS turbofan data (FD001).

Follows the loader of Bleistein et al. (github.com/LinusBleistein/signature_survival,
data_loader/load_NASA.py), without its torch dependency: train engines are
run to failure (events), test engines are censored at their last cycle;
features with >= 10 distinct values are standardized over train+test; all
engines are put on the common grid of observed times (forward-, then
back-filled), with time itself as the first feature.

    uv run python scripts/prepare_nasa.py --raw path/to/NASA   # dir with train_FD001.txt, test_FD001.txt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def load(raw: Path):
    feats = ["setting1", "setting2", "setting3"] + [f"s{i}" for i in range(1, 22)]
    cols = ["id", "times"] + feats
    tr = pd.read_csv(raw / "train_FD001.txt", sep=r"\s+", header=None, names=cols)
    te = pd.read_csv(raw / "test_FD001.txt", sep=r"\s+", header=None, names=cols)
    tr["tte"], tr["label"] = tr.groupby("id")["times"].transform("max"), 1
    te["tte"], te["label"] = te.groupby("id")["times"].transform("max"), 0
    te["id"] = te["id"] + tr["id"].max()
    keep = [c for c in feats if tr[c].nunique() >= 10]
    df = pd.concat([tr, te])[["id", "times", "tte", "label"] + keep]
    df[keep] = (df[keep] - df[keep].mean()) / df[keep].std()  # pandas default ddof=1, as in the original
    df["times"] = (df["times"] - 1) / 100
    df["tte"] = (df["tte"] - 1) / 100
    surv_times, surv_inds = df.drop_duplicates("id")[["tte", "label"]].values.T
    grid = np.concatenate((np.zeros(1), np.unique(df[["times", "tte"]].values)))
    ids = np.unique(df.id.values)
    X = np.zeros((len(ids), len(grid), 1 + len(keep)), dtype=np.single)
    for i, idx in enumerate(ids):
        g = pd.DataFrame({"id": float(idx), "times": grid})
        d = pd.merge(g, df[df.id == idx], how="left", on=["id", "times"]).ffill().bfill()
        X[i] = d[["times"] + keep].values
    labels = np.array([surv_times, surv_inds], dtype=np.single).T
    ts = (labels[:, 0] * 100).astype(int)   # float32 round trip, as in the original
    cs = (1 - labels[:, 1]).astype(int)
    return X, ts, cs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("data/NASA.h5"))
    a = ap.parse_args()
    X, ts, cs = load(a.raw)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(a.out, "w") as f:
        f["seqs"], f["ts"], f["cs"] = X, ts, cs
    print(f"{a.out}: seqs {X.shape}, events {int((cs == 0).sum())}, censored {int(cs.sum())}")


if __name__ == "__main__":
    main()
