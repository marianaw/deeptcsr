"""Freeze the winning configurations for the seed extension.

One config per (algorithm, dataset) per metric, chosen on MEAN validation
across seeds -- standard practice, see scripts/selection.py. Both protocols
are covered, each tuned on its own validation metric:
  TCSR protocol -> val_ci / val_bs
  DDH protocol  -> mean val_td_ci / val_td_bs over landmark cells
The union is written to results/final/selected_configs.csv and is what the
extra seeds run.
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from selection import pick_global

ALG = ["cox", "ddh", "inc_tc_cox", "inc_tc_ddh", "tc_cox", "tc_ddh"]
R = Path("results/final")

rows = []
flat = pd.read_csv(R / "all_runs.csv")
flat = flat[flat.algorithm.isin(ALG)]
for val, test, mx in [("val_ci", "test_ci", True), ("val_bs", "test_bs", False)]:
    s = pick_global(flat, ["algorithm", "dataset"], val, test, mx)
    rows.append(s[["algorithm", "dataset", "lambda_", "target_lr"]]
                .drop_duplicates().assign(basis=val))
land = pd.read_csv(R / "all_landmarks.csv")
land = land[(land.landmark > 0) & land.algorithm.isin(ALG)]
for val, test, mx in [("val_td_ci", "test_td_ci", True),
                      ("val_td_bs", "test_td_bs", False)]:
    s = pick_global(land, ["algorithm", "dataset"], val, test, mx)
    rows.append(s[["algorithm", "dataset", "lambda_", "target_lr"]]
                .drop_duplicates().assign(basis=val))

cfg = pd.concat(rows, ignore_index=True)
uniq = (cfg.groupby(["algorithm", "dataset", "lambda_", "target_lr"])["basis"]
        .apply(lambda b: "|".join(sorted(set(b)))).reset_index())
uniq.to_csv(R / "selected_configs.csv", index=False)
print(uniq.to_string(index=False))
print(f"\n{len(uniq)} configs x 20 seeds = {len(uniq)*20} runs")
