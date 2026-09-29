"""Metric-vs-tau curves: the stability/staleness tradeoff as a curve.

Inc-TCSR is not a separate method here -- it is the tau = 1.0 endpoint of the
same curve (its runs carry target_lr=1.0), so D-TCSR and Inc-TCSR lie on one
continuum. The untreated baseline is drawn as a horizontal reference, which
makes "does any tau beat doing nothing?" readable directly off the plot.

At each tau, lambda is re-selected on VALIDATION (per seed, per metric), so
each point is "the best you get at this tau if you tune lambda properly" --
not a single lambda held fixed across the sweep.

Bands are +/- 1 standard error over the 10 seeds.

  tau_curve_landmark.pdf  time-dependent C(t)/Brier at one landmark+horizon
  tau_curve_tcsr.pdf      global C-index/IBS under the TCSR (state-0) protocol
"""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

mpl.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "figure.dpi": 150})

DS = ["nasa", "scania", "churn_lastfm_months", "big_rw"]
LAB = {"nasa": "NASA", "scania": "Scania",
       "churn_lastfm_months": "LastFM", "big_rw": "Large-RW"}
FAM = {"LH": ("cox", "inc_tc_cox", "tc_cox", "#4472C4", "o"),
       "DDH": ("ddh", "inc_tc_ddh", "tc_ddh", "#1B7F5E", "s")}


def curve(df, fam, ds, val_col, test_col, maximize, extra_keys=()):
    """-> (taus, mean, sem, baseline_mean, baseline_sem)."""
    base, inc, dt, _, _ = FAM[fam]
    sub = df[(df.dataset == ds) & df.algorithm.isin([inc, dt])].dropna(
        subset=[val_col, test_col])
    if sub.empty:
        return None
    # lambda chosen ONCE per tau on mean validation across seeds
    m = sub.groupby(["target_lr", "lambda_"], as_index=False)[val_col].mean()
    idx = (m.groupby("target_lr")[val_col].idxmax() if maximize
           else m.groupby("target_lr")[val_col].idxmin())
    sel = sub.merge(m.loc[idx, ["target_lr", "lambda_"]],
                    on=["target_lr", "lambda_"], how="inner")
    g = sel.groupby("target_lr")[test_col]
    taus = np.array(sorted(sel.target_lr.unique()))
    mean = np.array([g.get_group(t).mean() for t in taus])
    sem = np.array([stats.sem(g.get_group(t)) if len(g.get_group(t)) > 1 else np.nan
                    for t in taus])
    b = df[(df.dataset == ds) & (df.algorithm == base)].dropna(subset=[test_col])
    bm = b[test_col].mean() if len(b) else np.nan
    bs = stats.sem(b[test_col]) if len(b) > 1 else np.nan
    return taus, mean, sem, bm, bs


def panel(ax, df, ds, val_col, test_col, maximize, ylabel, extra_keys=()):
    for fam, (_, _, _, colour, marker) in FAM.items():
        r = curve(df, fam, ds, val_col, test_col, maximize, extra_keys)
        if r is None:
            continue
        taus, m, s, bm, bs = r
        ax.plot(taus, m, marker=marker, ms=4, lw=1.6, color=colour, label=fam)
        ax.fill_between(taus, m - s, m + s, color=colour, alpha=0.18, lw=0)
        if np.isfinite(bm):
            ax.axhline(bm, color=colour, ls=":", lw=1.1, alpha=0.75)
        # tau = 1.0 IS Inc-TCSR -- mark the endpoint so it reads as such
        if 1.0 in set(taus):
            ax.plot([1.0], [m[list(taus).index(1.0)]], marker=marker, ms=8,
                    mfc="none", mec=colour, mew=1.6)
    ax.set_xscale("log")
    ax.set_xticks([0.05, 0.1, 0.25, 0.5, 1.0])
    ax.set_xticklabels(["0.05", "0.1", "0.25", "0.5", "1.0\n(Inc)"])
    ax.grid(axis="y", ls="--", alpha=0.35, lw=0.6)
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel)


def make(df, specs, out, suptitle, extra_keys=()):
    fig, axes = plt.subplots(2, len(DS), figsize=(3.4 * len(DS), 5.4),
                             squeeze=False)
    for row, (val_col, test_col, mx, ylab) in enumerate(specs):
        for col, ds in enumerate(DS):
            ax = axes[row][col]
            panel(ax, df, ds, val_col, test_col, mx,
                  ylab if col == 0 else "", extra_keys)
            if row == 0:
                ax.set_title(LAB[ds])
            if row == 1:
                ax.set_xlabel(r"target update rate $\tau$")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h + [plt.Line2D([], [], ls=":", color="0.4")],
               l + ["baseline (no TC)"], loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(suptitle, fontsize=11, y=0.995)
    fig.tight_layout(rect=[0, 0.045, 1, 0.975])
    for ext in ("pdf", "png"):
        fig.savefig(out.with_suffix("." + ext), bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/final", type=Path)
    a = ap.parse_args()
    flat = pd.read_csv(a.results / "all_runs.csv")
    land = pd.read_csv(a.results / "all_landmarks.csv")

    make(flat, [("val_ci", "test_ci", True, "C-Index"),
                ("val_bs", "test_bs", False, "IBS")],
         a.results / "tau_curve_tcsr",
         "Metric vs target update rate — TCSR protocol (state 0), "
         "$\\lambda$ re-tuned per $\\tau$, band = ±1 s.e.")

    # one landmark per dataset: the MEDIAN of the >0 landmarks, with its
    # median horizon -- enough subjects at risk to be stable, enough history
    # to be informative.
    rows = []
    for ds in DS:
        sub = land[(land.dataset == ds) & (land.landmark > 0)]
        lms = sorted(sub.landmark.unique())
        lm = lms[len(lms) // 2]
        hs = sorted(sub[sub.landmark == lm].horizon.unique())
        hz = hs[len(hs) // 2]
        rows.append(sub[(sub.landmark == lm) & (sub.horizon == hz)])
        print(f"  {LAB[ds]}: landmark t={lm}, horizon Δ={hz}")
    sel = pd.concat(rows)
    make(sel, [("val_td_ci", "test_td_ci", True, "C(t)-index"),
               ("val_td_bs_ipcw", "test_td_bs_ipcw", False, "Brier(t)")],
         a.results / "tau_curve_landmark",
         "Metric vs target update rate — DDH protocol at the median landmark, "
         "$\\lambda$ re-tuned per $\\tau$, band = ±1 s.e.")


if __name__ == "__main__":
    main()
