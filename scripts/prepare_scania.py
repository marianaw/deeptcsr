"""Build the SCANIA Component X sequence dataset.

Source: Swedish National Data Service, DOI 10.5878/bnh5-ka77 (version 3),
files ``train_operational_readouts.csv`` (107 cols: vehicle_id, time_step,
105 cumulative counters) and ``train_tte.csv`` (vehicle_id,
length_of_study_time_step, in_study_repair).

Representation follows the discrete-Markov requirement of TCSR:

* **Discrete step = readout index.** Each vehicle's readouts, ordered by
  ``time_step``, become steps 0, 1, 2, ... — the same convention PBC2/AIDS
  use for clinic visits. One step is "one readout", not a fixed duration.
* **The state is self-contained.** Step t carries the 105 counters as read
  at that time, so ``h(k | x_t)`` conditions on a single state rather than
  on history. No deltas or lags are added, which would make the state a
  function of two time points.
* **Nothing follows the event.** Verified in the raw data: every readout
  lies at or before ``length_of_study_time_step`` (for repaired vehicles the
  last readout precedes the repair, median gap 2.0). Sequences therefore end
  at the event by construction; we only truncate at the horizon.

Normalisation: the counters are monotone non-decreasing and span ~4 orders
of magnitude (max 3.4e10), so each feature is ``log1p``-transformed and then
standardised over observed (non-padded) readouts. Missing cells (0.3%) map
to the feature mean, i.e. 0 after standardisation.

Labels use the project convention ``ts = n_steps - censored``. Vehicles whose
sequence exceeds the horizon are administratively censored at the horizon.

Run:
    uv run python scripts/prepare_scania.py            # 5,000-vehicle subsample
    uv run python scripts/prepare_scania.py --all      # all 23,550 vehicles
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/scania_component_x_v3")
OUT = Path("data/scania")
SUBSAMPLE = Path("results/scania_component_x/subsample_vehicle_ids.csv")


def load_readouts(keep_ids: set[int] | None, chunksize: int = 500_000):
    """Stream the 1.2 GB readout file, keeping only the requested vehicles."""
    parts = []
    for chunk in pd.read_csv(RAW / "train_operational_readouts.csv",
                             chunksize=chunksize):
        if keep_ids is not None:
            chunk = chunk[chunk.vehicle_id.isin(keep_ids)]
        if len(chunk):
            parts.append(chunk)
    return pd.concat(parts, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=100,
                    help="max steps kept; p95 sequence length is 97")
    ap.add_argument("--all", action="store_true",
                    help="use all 23,550 vehicles instead of the 5,000 subsample")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    tte = pd.read_csv(RAW / "train_tte.csv")
    keep = None
    if not args.all:
        keep = set(pd.read_csv(SUBSAMPLE).vehicle_id.tolist())
        tte = tte[tte.vehicle_id.isin(keep)]

    logs = load_readouts(keep).sort_values(["vehicle_id", "time_step"],
                                           kind="stable")
    cols = [c for c in logs.columns if c not in ("vehicle_id", "time_step")]

    # log1p + standardise over observed readouts (counters are non-negative
    # and span ~4 orders of magnitude, so raw values are unusable as-is).
    vals = np.log1p(logs[cols].to_numpy(dtype=np.float64))
    mean = np.nanmean(vals, axis=0)
    std = np.nanstd(vals, axis=0)
    std = np.where((std < 1e-12) | ~np.isfinite(std), 1.0, std)
    vals = (np.where(np.isfinite(vals), vals, mean) - mean) / std
    logs = pd.DataFrame(vals, columns=cols).assign(
        vehicle_id=logs.vehicle_id.to_numpy())

    H = args.horizon
    order = tte.vehicle_id.to_numpy()
    idx = {v: i for i, v in enumerate(order)}
    seqs = np.zeros((len(order), H, len(cols)), dtype=np.float32)
    n_steps = np.zeros(len(order), dtype=int)

    for v, g in logs.groupby("vehicle_id", sort=False):
        i = idx[v]
        a = g[cols].to_numpy(dtype=np.float32)[:H]
        seqs[i, :len(a)] = a
        n_steps[i] = len(a)

    event = (tte.in_study_repair.to_numpy() > 0)
    truncated = n_steps >= H          # administratively censored at the horizon
    cs = (~event) | truncated
    ts = n_steps - cs.astype(int)     # project convention: ts = count - censored
    ts = np.clip(ts, 1, H)

    out = args.out or (OUT / f"scania-seqs-h{H}{'-full' if args.all else ''}.pkl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump({"seqs": seqs, "ts": ts, "cs": cs, "cols": cols}, f)

    print(f"{out}: seqs {seqs.shape}")
    print(f"  events {int((~cs).sum())}  censored {int(cs.sum())} "
          f"({cs.mean()*100:.1f}%)  truncated-at-horizon {int(truncated.sum())}")
    print(f"  ts: min {ts.min()} median {int(np.median(ts))} max {ts.max()}"
          f"  | events/bin {(~cs).sum()/H:.1f}")


if __name__ == "__main__":
    main()
