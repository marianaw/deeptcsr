"""Generate the LargeRW dataset (data/bigrw-seqs.pkl).

Gauss-Markov random walk of the tdsurv random-walk notebook (Maystre & Russo),
scaled up: 10,000 sequences of 49 random-walk features plus a constant,
horizon 99. x_0 ~ N(0, I), x_{l+1} = x_l + N(0, 0.5^2 I); at each state the
sequence churns with probability sigmoid(theta . x_l - 8). Sequences that
reach the horizon are censored (~22%).

    uv run python scripts/prepare_large_rw.py
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
from scipy.special import expit as sigmoid


def generate(n, horizon, n_dims, bias, sigma, rng):
    thetas = rng.normal(size=n_dims)
    seqs = np.zeros((n, horizon + 1, n_dims + 1))
    ts = np.ones(n, dtype=int)
    cs = np.zeros(n, dtype=bool)
    for i in range(n):
        seqs[i, 0] = np.append(rng.normal(size=n_dims), 1)
        for j in range(horizon):
            if rng.uniform() < sigmoid(seqs[i, j, :-1] @ thetas + bias):
                break
            ts[i] += 1
            seqs[i, j + 1] = np.append(seqs[i, j, :-1] + sigma * rng.normal(size=n_dims), 1)
        if ts[i] > horizon:
            ts[i], cs[i] = horizon, True
    return seqs.astype(np.float32), ts, cs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/bigrw-seqs.pkl"))
    a = ap.parse_args()
    seqs, ts, cs = generate(10_000, 99, 49, -8.0, 0.5, np.random.default_rng(0))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "wb") as f:
        pickle.dump({"seqs": seqs, "ts": ts, "cs": cs}, f)
    print(f"{a.out}: seqs {seqs.shape}, censored {cs.mean():.1%}")


if __name__ == "__main__":
    main()
