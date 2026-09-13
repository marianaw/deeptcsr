"""Build the small-dataset pickles (PBC2, AIDS, small random walk).

Replicates the preprocessing of the tdsurv notebooks
(`pbc2-exploratory-analysis.ipynb`, `aids-exploratory-analysis.ipynb`,
`random-walk-final.ipynb`) so the resulting ``*-seqs.pkl`` files match the
ones used by the original TCSR paper (Maystre & Russo).

Inputs (in ``data/small_datasets/raw/``):
  - ``pbc.rda``  — from the R ``survival`` package (contains ``pbcseq``,
    the same 1,945 visit rows as Mayo's DOC-10026794).
  - ``aids.rda`` — from the R ``JM`` package.

Run with: ``uv run --with pyreadr python scripts/prepare_small_data.py``
"""

from __future__ import annotations

import os
import pickle

import numpy as np
import pandas as pd
import pyreadr
from scipy.special import expit as sigmoid

RAW = "data/small_datasets/raw"
OUT = "data/small_datasets"


def _to_sequences(final: pd.DataFrame, key: str, cols: list[str]):
    """Dense (n, max_len, d) array from long-format visits, tdsurv-style."""
    agg = final.groupby(key)
    n = len(agg)
    d = len(cols)
    m = agg[cols[0]].count().max()
    seqs = np.zeros((n, m, d))
    for j, col in enumerate(cols):
        for i, seq in enumerate(agg[col].apply(list)):
            seqs[i, : len(seq), j] = seq
    return seqs


def prepare_pbc2():
    df = pyreadr.read_r(os.path.join(RAW, "pbc.rda"))["pbcseq"]
    # Rename/recode to the Mayo DOC-10026794 column conventions used by tdsurv.
    final = pd.DataFrame({
        "id": df["id"].astype(int),
        "label": df["status"].astype(int),           # 0 alive, 1 transplant, 2 dead
        "drug": (df["trt"] == 1).astype(float),      # 1 = D-penicillamine
        "age": df["age"].astype(float),
        "sex": (df["sex"] == "f").astype(float),
        "ascites": df["ascites"].astype(float),
        "hepatomegaly": df["hepato"].astype(float),
        "spiders": df["spiders"].astype(float),
        "edema": df["edema"].astype(float),
        "serBilir": df["bili"].astype(float),
        "serChol": df["chol"].astype(float),
        "albumin": df["albumin"].astype(float),
        "alkaline": df["alk.phos"].astype(float),
        "SGOT": df["ast"].astype(float),
        "platelets": df["platelet"].astype(float),
        "prothrombin": df["protime"].astype(float),
        "histologic": df["stage"].astype(float),
    })

    for col in ["ascites", "hepatomegaly", "spiders", "serChol",
                "alkaline", "platelets"]:
        final.loc[final[col].isna(), col] = final[col].median()

    final["age"] = (final["age"] - final["age"].mean()) / final["age"].std()
    for col in ["serBilir", "serChol", "albumin", "alkaline", "SGOT",
                "platelets", "prothrombin"]:
        final[col] = np.log1p(final[col]) - np.mean(np.log1p(final[col]))

    agg = final.groupby("id")["label"]
    cs = (agg.first() != 2).values
    ts = agg.count().values - cs.astype(int)

    final = final.drop("label", axis=1)
    cols = [c for c in final.columns if c != "id"]
    seqs = _to_sequences(final, "id", cols)
    _dump("pbc-seqs.pkl", seqs, ts, cs, cols)


def prepare_aids():
    df = pyreadr.read_r(os.path.join(RAW, "aids.rda"))["aids"]
    final = pd.DataFrame({
        "patient": df["patient"].astype(int),
        "death": df["death"].astype(bool),
        "drug": (df["drug"] == "ddC").astype(float),
        "gender": (df["gender"] == "female").astype(float),
        "prevOI": (df["prevOI"] == "AIDS").astype(float),
        "AZT": (df["AZT"] == "intolerance").astype(float),
        "CD4": df["CD4"].astype(float),
    }).sort_values(["patient"], kind="stable")

    vals = np.sqrt(final["CD4"])
    final["sqrtCD4"] = vals - vals.mean()

    agg = final.groupby("patient")["death"]
    cs = (~agg.first()).values
    ts = agg.count().values - cs.astype(int)

    cols = ["drug", "gender", "prevOI", "AZT", "sqrtCD4"]
    seqs = _to_sequences(final, "patient", cols)
    _dump("aids-seqs.pkl", seqs, ts, cs, cols)


def prepare_rw(n_samples=1000, n_dims=20, horizon=10, seed=0):
    """Small Gauss-Markov random walk, per tdsurv random-walk-final.ipynb."""
    rng = np.random.default_rng(seed=seed)
    thetas = rng.normal(size=n_dims)
    bias = -3.0
    mat = np.eye(n_dims)
    sigma, sigma0 = 0.5, 1.0

    seqs = np.zeros((n_samples, horizon + 1, n_dims + 1))
    ts = np.zeros(n_samples, dtype=int)
    cs = np.zeros(n_samples, dtype=bool)
    for i in range(n_samples):
        seqs[i, 0] = np.append(sigma0 * rng.normal(size=n_dims), 1)
        ts[i] = 1
        for j in range(horizon):
            p = sigmoid(np.dot(seqs[i, j, :-1], thetas) + bias)
            if rng.uniform() < p:
                break
            ts[i] += 1
            seqs[i, j + 1] = np.append(
                np.dot(mat, seqs[i, j + 1, :-1]) + sigma * rng.normal(size=n_dims),
                1,
            )
        if ts[i] > horizon:
            ts[i] = horizon
            cs[i] = True
    cols = [f"x{k}" for k in range(n_dims)] + ["bias"]
    _dump("rw-seqs.pkl", seqs, ts, cs, cols)


def _dump(fname, seqs, ts, cs, cols):
    path = os.path.join(OUT, fname)
    with open(path, "wb") as f:
        pickle.dump({"seqs": seqs, "ts": ts, "cs": cs, "cols": cols}, f)
    print(f"{fname}: seqs {seqs.shape}, events {np.sum(~cs)}, "
          f"censored {np.sum(cs)}, ts in [{ts.min()}, {ts.max()}]")


if __name__ == "__main__":
    prepare_pbc2()
    prepare_aids()
    prepare_rw()
