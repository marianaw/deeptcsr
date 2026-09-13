"""Fitted TCSR (Maystre & Russo) baseline on the small datasets.

Uses the reference implementation from the tdsurv repo (CoxPH model, fitted
with Newton-CG on iteratively recomputed pseudo-targets, lambda_=0 = pure
one-step bootstrap). Splits, seeds and metrics are identical to the
scripts/run.py protocol so results are directly comparable.

Each l2 value in the grid is written to its own run directory (the
``target_lr_<l2>`` path slot doubles as the l2 axis for tcsr_fitted runs) so
the aggregation step can select l2 on validation per metric, exactly like
tau for D-TCSR.

Usage:
    uv run python scripts/run_fitted_tcsr.py --dataset aids --seed 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import yaml

TDSURV_LIB = os.environ.get(
    "TDSURV_LIB",
    os.path.join(os.path.dirname(__file__), "..", "..", "tdsurv", "lib"),
)
sys.path.insert(0, TDSURV_LIB)

from tdsurv import CoxPH  # noqa: E402
from tdsurv.utils import unroll  # noqa: E402

from survan.data import load_dataset, train_val_test_split  # noqa: E402
from survan.losses import median_survival_time  # noqa: E402
from survan.metrics import (concordance_index, concordance_index_ipcw,  # noqa: E402
                            integrated_brier_score_ipcw,
                            integrated_brier_score_np)

L2_GRID = (0.0, 0.01, 0.1, 1.0)
N_ITERS = 100


def _evaluate(model, x0, ts, cs, train_ts, train_cs):
    surv = np.asarray(model.survival_curve(x0))
    scores = np.asarray(median_survival_time(surv))
    ci = float(concordance_index(scores, ts, cs))
    ci_ipcw = concordance_index_ipcw(scores, ts, cs, train_ts, train_cs)
    ibs = integrated_brier_score_np(surv[:, 1:], ts, cs)
    ibs_ipcw = integrated_brier_score_ipcw(surv[:, 1:], ts, cs,
                                           train_ts, train_cs)
    return ci, ci_ipcw, ibs, ibs_ipcw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--output-dir", default="outputs")
    ap.add_argument("--test-seed", type=int, default=None,
                    help="Fix the test split with this seed (learning-curve "
                         "protocol); --seed then only reshuffles train/val.")
    ap.add_argument("--n-train", type=int, default=None,
                    help="Truncate the shuffled train pool to this many "
                         "sequences (requires --test-seed).")
    args = ap.parse_args()

    with open(f"configs/dataset/{args.dataset}.yaml") as f:
        ds = yaml.safe_load(f)
    data_path = ds["data_path"].replace("${data_root}", args.data_root)

    def _run_dir(l2):
        return os.path.join(
            args.output_dir, "tcsr_fitted", ds["name"], "linear",
            "lambda_0.0", f"target_lr_{l2}", "landmark_True",
            f"seed_{args.seed}")

    if all(os.path.exists(os.path.join(_run_dir(l2), "results_test.json"))
           for l2 in L2_GRID):
        print(f"skip (already done): {_run_dir(L2_GRID[0])} …")
        return

    seqs, ts, cs, *_ = load_dataset(
        ds["name"], landmark=ds["landmark"], compute_targets=False,
        kwargs={"data_path": data_path, "horizon": ds["horizon"]})
    if args.test_seed is not None:
        s1 = train_val_test_split({"X": seqs}, ts, cs, seed=args.test_seed,
                                  test_size=0.2, val_size=None,
                                  stratify=ds["stratify"])
        rest = s1["train"]
        s2 = train_val_test_split({"X": rest["X"]}, rest["ts"], rest["cs"],
                                  seed=args.seed, test_size=0.25,
                                  val_size=None, stratify=ds["stratify"])
        tr, va, te = s2["train"], s2["test"], s1["test"]
        if args.n_train is not None:
            tr = {k: v[:args.n_train] for k, v in tr.items()}
    else:
        splits = train_val_test_split(
            {"X": seqs}, ts, cs, seed=args.seed,
            test_size=0.2, val_size=0.2, stratify=ds["stratify"])
        tr, va, te = splits["train"], splits["val"], splits["test"]
    horizon = ds["horizon"]
    d = seqs.shape[-1]
    seqs_u, ts_u, cs_u = unroll(tr["X"].astype(np.float64), tr["ts"],
                                tr["cs"].astype(bool))

    for l2 in L2_GRID:
        model = CoxPH(horizon=horizon, n_feats=d)
        model.fit(seqs_u, ts_u, cs_u, lambda_=0.0, n_iters=N_ITERS, l2=l2)
        val = _evaluate(model, va["X"][:, 0].astype(np.float64),
                        va["ts"], va["cs"].astype(bool), tr["ts"], tr["cs"])
        test = _evaluate(model, te["X"][:, 0].astype(np.float64),
                         te["ts"], te["cs"].astype(bool), tr["ts"], tr["cs"])
        run_dir = _run_dir(l2)
        os.makedirs(run_dir, exist_ok=True)
        for split, (ci, ci_ipcw, ibs, ibs_ipcw) in (("val", val),
                                                    ("test", test)):
            with open(os.path.join(run_dir, f"results_{split}.json"), "w") as f:
                json.dump({"split": split, "ci": float(ci), "bs": float(ibs),
                           "ci_ipcw": float(ci_ipcw),
                           "bs_ipcw": float(ibs_ipcw), "l2": l2}, f)
        print(f"l2={l2}: val ci={val[0]:.4f} bs={val[2]:.4f}  "
              f"test ci={test[0]:.4f} bs={test[2]:.4f} → {run_dir}")


if __name__ == "__main__":
    main()
