"""Landmark x horizon "wins" per dataset: baseline vs Inc-TCSR vs D-TCSR.

Each (landmark > 0, horizon) cell is one contest. Configs are chosen exactly
as in make_final_plots.py (one per algorithm/dataset, on mean validation over
the tuning seeds, reported over all seeds). The winner of a cell is the arm
with the best mean test metric; the win is SIGNIFICANT when the winner beats
BOTH other arms in a paired t-test over seeds at p < ALPHA.

Writes results/final/win_counts.csv and win_counts.md.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy import stats

from selection import pick_global

ALPHA = 0.05
DS = ["nasa", "scania", "churn_lastfm_months", "big_rw"]
LAB = {"nasa": "NASA", "scania": "Scania",
       "churn_lastfm_months": "LastFM", "big_rw": "Large-RW"}
FAM = {"LH": ("cox", "inc_tc_cox", "tc_cox"),
       "DDH": ("ddh", "inc_tc_ddh", "tc_ddh")}
ARMS = ["baseline", "Inc-TCSR", "D-TCSR"]
METRICS = {"C(t)": ("val_td_ci", "test_td_ci", True),
           "Brier(t) IPCW": ("val_td_bs_ipcw", "test_td_bs_ipcw", False)}


def cell_winner(cell: pd.DataFrame, algos, test_col, higher):
    per = {a: cell[cell.algorithm == a].set_index("seed")[test_col].dropna()
           for a in algos}
    if any(len(v) < 2 for v in per.values()):
        return None, False
    means = {a: v.mean() for a, v in per.items()}
    win = max(means, key=means.get) if higher else min(means, key=means.get)
    sig = True
    for other in algos:
        if other == win:
            continue
        common = per[win].index.intersection(per[other].index)
        if len(common) < 2 or stats.ttest_rel(per[win][common],
                                              per[other][common]).pvalue >= ALPHA:
            sig = False
    return ARMS[list(algos).index(win)], sig


def pairwise(cell: pd.DataFrame, a: str, b: str, test_col, higher):
    """-> (a better than b?, p-value) over seeds present for both, or None."""
    x = cell[cell.algorithm == a].set_index("seed")[test_col].dropna()
    y = cell[cell.algorithm == b].set_index("seed")[test_col].dropna()
    common = x.index.intersection(y.index)
    if len(common) < 2:
        return None
    d = (x[common] - y[common]).mean()
    return (d > 0 if higher else d < 0), stats.ttest_rel(x[common], y[common]).pvalue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/final", type=Path)
    a = ap.parse_args()
    land = pd.read_csv(a.results / "all_landmarks.csv")
    land = land[land.landmark > 0]
    rows = []
    for metric, (val_col, test_col, higher) in METRICS.items():
        sel = pick_global(land, ["algorithm", "dataset"], val_col, test_col, higher)
        for fam, algos in FAM.items():
            for ds in DS:
                sub = sel[(sel.dataset == ds) & sel.algorithm.isin(algos)]
                counts = {arm: [0, 0] for arm in ARMS}
                vs = {"better": 0, "better sig": 0, "worse": 0, "worse sig": 0}
                n = 0
                for _, cell in sub.groupby(["landmark", "horizon"]):
                    pw = pairwise(cell, algos[2], algos[1], test_col, higher)
                    if pw is not None:
                        key = "better" if pw[0] else "worse"
                        vs[key] += 1
                        vs[f"{key} sig"] += pw[1] < ALPHA
                    win, sig = cell_winner(cell, algos, test_col, higher)
                    if win is None:
                        continue
                    n += 1
                    counts[win][0] += 1
                    counts[win][1] += sig
                rows.append({"metric": metric, "family": fam, "dataset": LAB[ds],
                             "cells": n,
                             **{f"{arm} wins": counts[arm][0] for arm in ARMS},
                             **{f"{arm} sig": counts[arm][1] for arm in ARMS},
                             **{f"D vs Inc {k}": v for k, v in vs.items()}})
    df = pd.DataFrame(rows)
    df.to_csv(a.results / "win_counts.csv", index=False)

    lines = [f"Cells = landmark (>0) x horizon. Entry = wins (significant wins: "
             f"winner beats both other arms, paired t-test over seeds, p<{ALPHA}).",
             f"Last column: cells where D-TCSR is better / worse than Inc-TCSR "
             f"(in parentheses: significant, paired t-test, p<{ALPHA}).", ""]
    for metric in METRICS:
        for fam in FAM:
            t = df[(df.metric == metric) & (df.family == fam)]
            lines += [f"### {fam} - {metric}", "",
                      "| dataset | cells | " + " | ".join(ARMS) + " | D-TCSR vs Inc-TCSR: better · worse |",
                      "|---|---|" + "---|" * len(ARMS) + "---|"]
            for r in t.to_dict("records"):
                cells = " | ".join(f"{r[f'{arm} wins']} ({r[f'{arm} sig']})" for arm in ARMS)
                vs = (f"{r['D vs Inc better']} ({r['D vs Inc better sig']}) · "
                      f"{r['D vs Inc worse']} ({r['D vs Inc worse sig']})")
                lines.append(f"| {r['dataset']} | {r['cells']} | {cells} | {vs} |")
            tot = {arm: (t[f"{arm} wins"].sum(), t[f"{arm} sig"].sum()) for arm in ARMS}
            v = {k: t[f"D vs Inc {k}"].sum() for k in ("better", "better sig", "worse", "worse sig")}
            lines.append(f"| **total** | {t.cells.sum()} | "
                         + " | ".join(f"{w} ({s})" for w, s in tot.values())
                         + f" | {v['better']} ({v['better sig']}) · {v['worse']} ({v['worse sig']}) |")
            lines.append("")
    (a.results / "win_counts.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
