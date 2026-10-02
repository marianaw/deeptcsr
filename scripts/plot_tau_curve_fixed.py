"""Clean tau curves: ONE configuration, tau varied, everything else held.

Two defects in the earlier version are fixed here:
  * lambda was re-selected at every tau, so the line mixed lambda effects
    with tau effects and kinked wherever the winning lambda jumped
    (NASA: 0.95 -> 0.1 -> 0.95). Here lambda is PINNED to the winning value.
  * seeds differed per tau (10 at tau=0.5/0.75, 30 elsewhere), so points sat
    on different noise floors. Here only seeds present at EVERY tau are used.

Cell choice: per dataset the (landmark, horizon) where D-TCSR most improves
on Inc-TCSR -- a cell the method wins, which is where the stability/staleness
tradeoff should be visible as a curve rather than a wash.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib as mpl, matplotlib.pyplot as plt
import numpy as np, pandas as pd
from scipy import stats

mpl.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "figure.dpi": 150})
DS = ["nasa", "scania", "churn_lastfm_months", "big_rw"]
LAB = {"nasa": "NASA", "scania": "Scania",
       "churn_lastfm_months": "LastFM", "big_rw": "Large-RW"}
FAM = {"LH": ("cox", "inc_tc_cox", "tc_cox", "#4472C4", "o"),
       "DDH": ("ddh", "inc_tc_ddh", "tc_ddh", "#1B7F5E", "s")}


def best_cell(land, ds, inc, dt, col):
    """(landmark, horizon) with the largest D-TCSR - Inc-TCSR gap."""
    sub = land[(land.dataset == ds) & (land.landmark > 0)]
    best, gap = None, -np.inf
    for k, g in sub.groupby(["landmark", "horizon"]):
        a = g[g.algorithm == inc][col].mean()
        b = g[g.algorithm == dt][col].mean()
        if np.isfinite(a) and np.isfinite(b) and b - a > gap:
            best, gap = k, b - a
    return best


def series(land, ds, fam, cell, val_col, test_col, maximize):
    base, inc, dt, colour, marker = FAM[fam]
    lm, hz = cell
    sub = land[(land.dataset == ds) & (land.landmark == lm) &
               (land.horizon == hz)].dropna(subset=[val_col, test_col])
    tc = sub[sub.algorithm == dt]
    if tc.empty:
        return None
    # pin lambda: the single winner on mean validation across all taus
    ml = tc.groupby("lambda_")[val_col].mean()
    lam = ml.idxmax() if maximize else ml.idxmin()
    tc = tc[tc.lambda_ == lam]
    inc_rows = sub[(sub.algorithm == inc) & (sub.lambda_ == lam)]
    pts = pd.concat([tc, inc_rows])
    # equalise seeds: keep only seeds present at EVERY tau
    per = pts.groupby("target_lr").seed.apply(set)
    common = set.intersection(*per) if len(per) else set()
    pts = pts[pts.seed.isin(common)]
    taus = np.array(sorted(pts.target_lr.unique()))
    g = pts.groupby("target_lr")[test_col]
    m = np.array([g.get_group(t).mean() for t in taus])
    s = np.array([stats.sem(g.get_group(t)) for t in taus])
    b = sub[sub.algorithm == base][test_col]
    return taus, m, s, (b.mean() if len(b) else np.nan), lam, len(common)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/final", type=Path)
    a = ap.parse_args()
    land = pd.read_csv(a.results / "all_landmarks.csv")
    fig, axes = plt.subplots(2, len(DS), figsize=(3.4 * len(DS), 5.6), squeeze=False)
    for col, ds in enumerate(DS):
        cell = best_cell(land, ds, "inc_tc_cox", "tc_cox", "test_td_ci")
        for row, (val_col, test_col, mx, ylab) in enumerate(
                [("val_td_ci", "test_td_ci", True, "C(t)-index"),
                 ("val_td_bs_ipcw", "test_td_bs_ipcw", False, "Brier(t)")]):
            ax = axes[row][col]
            note = []
            for fam in FAM:
                r = series(land, ds, fam, cell, val_col, test_col, mx)
                if r is None:
                    continue
                taus, m, s, bm, lam, nseed = r
                _, _, _, colour, marker = FAM[fam]
                ax.plot(taus, m, marker=marker, ms=4, lw=1.6, color=colour,
                        label=f"{fam} (λ={lam:g})")
                ax.fill_between(taus, m - s, m + s, color=colour, alpha=0.18, lw=0)
                if np.isfinite(bm):
                    ax.axhline(bm, color=colour, ls=":", lw=1.1, alpha=0.7)
                note.append(nseed)
            ax.set_xscale("log")
            ax.set_xticks([0.001, 0.01, 0.1, 1.0])
            ax.set_xticklabels(["0.001", "0.01", "0.1", "1.0\n(Inc)"])
            ax.grid(axis="y", ls="--", alpha=0.35, lw=0.6); ax.set_axisbelow(True)
            ax.legend(fontsize=7, loc="best")
            if col == 0:
                ax.set_ylabel(ylab)
            if row == 0:
                ax.set_title(f"{LAB[ds]}\nt={cell[0]}, Δ={cell[1]}  ({min(note)} seeds)")
            else:
                ax.set_xlabel(r"target update rate $\tau$")
    fig.suptitle("Metric vs $\\tau$ at a winning cell — $\\lambda$ pinned, "
                 "seeds equalised, band = ±1 s.e.", fontsize=11, y=0.995)
    fig.tight_layout(rect=[0, 0.01, 1, 0.975])
    for ext in ("pdf", "png"):
        fig.savefig(a.results / f"tau_curve_fixed.{ext}", bbox_inches="tight")
    print("wrote", a.results / "tau_curve_fixed.pdf")


if __name__ == "__main__":
    main()
