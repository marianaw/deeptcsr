"""Seed extension for the small-data benchmark: seeds 10..29 for every
configuration selected (on seeds 0..9) by scripts/small_data_table.py, under
either metric. Same commands as scripts/run_small_data.sh; finished runs are
skipped by run.py / run_fitted_tcsr.py, so this is safe to re-run.

    uv run python scripts/run_small_seed_extension.py [--root outputs_small_v2] [-j 4]
"""
from __future__ import annotations

import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

SEEDS = range(10, 30)
TEST_SEED = 1234


def commands(sel: pd.DataFrame, root: str, data_root: str) -> list[str]:
    cmds = set()
    for r in sel.itertuples():
        n = int(r.n_train)
        out = f"{root}/ntrain_{n}"
        if r.algorithm == "tcsr_fitted":  # one call fits every l2
            for s in SEEDS:
                cmds.add(f"uv run python scripts/run_fitted_tcsr.py --dataset {r.dataset} "
                         f"--seed {s} --test-seed {TEST_SEED} --n-train {n} "
                         f"--data-root {data_root} --output-dir {out}")
            continue
        algo = {"cox": "algorithm=cox backbone=linear algorithm.loss_norm=mean",
                "inc_tcsr": "algorithm=d_tcsr algorithm.name=inc_tcsr algorithm.target_lr=1.0",
                "d_tcsr": f"algorithm=d_tcsr algorithm.target_lr={r.target_lr}"}[r.algorithm]
        cmds.add(f"uv run python scripts/run.py -m dataset={r.dataset} {algo} "
                 f"test_seed={TEST_SEED} n_train={n} output_dir={out} "
                 f"'seed=range({SEEDS.start},{SEEDS.stop})' data_root={data_root}")
    return sorted(cmds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/small_data", type=Path)
    ap.add_argument("--root", default="outputs_small_v2")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("-j", type=int, default=4)
    a = ap.parse_args()
    sel = pd.concat([pd.read_csv(a.results / f) for f in
                     ("selected_val_ci.csv", "selected_val_ibs.csv")])
    sel = sel[["algorithm", "dataset", "n_train", "lambda_", "target_lr"]].drop_duplicates()
    cmds = commands(sel, a.root, a.data_root)
    print(f"{len(sel)} selected configs -> {len(cmds)} commands", flush=True)

    def run(c):
        r = subprocess.run(c, shell=True, capture_output=True)
        if r.returncode:
            print(f"FAILED: {c}", flush=True)

    with ThreadPoolExecutor(a.j) as ex:
        list(ex.map(run, cmds))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
