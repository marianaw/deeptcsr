"""Publication figures for the large benchmarks.

Two families, two protocols, one visual language:

  summary_LH.pdf        Logistic Hazard family, TCSR protocol (state-0
                        read-out): C-index and IBS per dataset.
  summary_DDH.pdf       Dynamic-DeepHit family, DDH protocol: time-dependent
                        C(t)-index across the landmark x horizon quantile grid.
  summary_LH_landmarks.pdf  the LH family under the DDH protocol, for the
                        appendix -- same grid, so the two are comparable.

Style follows the existing summary_COX.pdf: grouped bars, SEM whiskers,
dashed y-grid, significance asterisks for D-TCSR vs Inc-TCSR.

Colour: three series need three hues. Blue is kept from the original figure;
orange and green are Okabe-Ito values chosen for colour-vision deficiency.
Each arm also carries a distinct hatch, so identity never rests on colour
alone (the node palette validator was unavailable here, which makes the
secondary encoding load-bearing rather than decorative).
"""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

mpl.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.dpi": 150,
})

DS = ["nasa", "scania", "churn_lastfm_months", "big_rw"]
LAB = {"nasa": "NASA", "scania": "Scania",
       "churn_lastfm_months": "LastFM", "big_rw": "Large-RW"}
# family -> (baseline, inc, d-tcsr) algorithm keys
FAM = {"LH": ("cox", "inc_tc_cox", "tc_cox"),
       "DDH": ("ddh", "inc_tc_ddh", "tc_ddh")}
ARMS = ["baseline", "Inc-TCSR", "D-TCSR"]
STYLE = [("#4472C4", None), ("#E69F00", "//"), ("#1B7F5E", "xx")]
STAR = "#C0392B"


def stars(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else ""


def grouped_bars(ax, groups, means, sems, sig, ylabel, title, xlabels):
    """means/sems: (n_groups, 3). sig: per-group asterisk string for D-TCSR."""
    x = np.arange(len(groups))
    w = 0.26
    for i, (arm, (c, h)) in enumerate(zip(ARMS, STYLE)):
        off = (i - 1) * w
        ax.bar(x + off, means[:, i], w, yerr=sems[:, i], capsize=2.5,
               color=c, hatch=h, edgecolor="white", linewidth=0.6,
               label=arm, error_kw={"elinewidth": 0.8, "capthick": 0.8})
    for g, s in enumerate(sig):
        if s:
            top = means[g, 2] + (sems[g, 2] if np.isfinite(sems[g, 2]) else 0)
            ax.text(x[g] + w, top + 0.012 * max(1e-9, np.nanmax(means)),
                    s, ha="center", va="bottom", color=STAR, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.35, linewidth=0.6)
    ax.set_axisbelow(True)


def agg(subs, algos, metric, higher_better=True):
    """subs: list of per-group DataFrames.
    -> means (n,3), sems (n,3), markers (n,) for D-TCSR vs Inc-TCSR.

    Markers are DIRECTION-AWARE: asterisks mean D-TCSR is significantly
    BETTER; a significant loss is marked "†" instead, so a star can never be
    read as a win when the metric is one where lower is better."""
    m = np.full((len(subs), 3), np.nan)
    s = np.full((len(subs), 3), np.nan)
    sig = []
    for r, sub in enumerate(subs):
        per = {}
        for c, a in enumerate(algos):
            g = sub[sub.algorithm == a]
            if len(g):
                m[r, c] = g[metric].mean()
                s[r, c] = stats.sem(g[metric]) if len(g) > 1 else np.nan
                per[a] = g.set_index("seed")[metric]
        inc, dt = algos[1], algos[2]
        mark = ""
        if inc in per and dt in per:
            common = per[inc].index.intersection(per[dt].index)
            if len(common) > 1:
                diff = (per[dt].loc[common] - per[inc].loc[common]).mean()
                pv = stats.ttest_rel(per[dt].loc[common], per[inc].loc[common]).pvalue
                better = diff > 0 if higher_better else diff < 0
                mark = stars(pv) if better else ("\u2020" if pv < 0.05 else "")
        sig.append(mark)
    return m, s, sig


def pick(df, keys, val_col, test_col, maximize):
    d = df.dropna(subset=[val_col, test_col])
    idx = (d.groupby(keys)[val_col].idxmax() if maximize
           else d.groupby(keys)[val_col].idxmin())
    return d.loc[idx]


def fig_tcsr(flat, family, out):
    algos = FAM[family]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
    for ax, (metric, val_col, mx, ylab, ttl) in zip(axes, [
            ("test_ci", "val_ci", True, "C-Index", "C-Index  (higher is better)"),
            ("test_bs", "val_bs", False, "IBS", "Integrated Brier Score  (lower is better)")]):
        sel = pick(flat, ["algorithm", "dataset", "seed"], val_col, metric, mx)
        subs = [sel[sel.dataset == d] for d in DS]
        m, s, sig = agg(subs, algos, metric, higher_better=mx)
        grouped_bars(ax, DS, m, s, sig, ylab, ttl, [LAB[d] for d in DS])
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, [f"{family} {a}" for a in l], loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.04))
    fig.text(0.985, -0.03,
             "* p<0.05   ** p<0.01   *** p<0.001   \u2020 D-TCSR worse (p<0.05)",
             fontsize=7.5, ha="right", color="0.35")
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    for ext in ("pdf", "png"):
        fig.savefig(out.with_suffix("." + ext), bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def fig_landmarks(land, family, out, metric="ci"):
    """metric="ci" -> time-dependent C(t)-index; "bs" -> time-dependent Brier.
    Each is selected on its OWN validation counterpart, never on the other."""
    algos = FAM[family]
    val_col, test_col, mx, ylab, what = (
        ("val_td_ci", "test_td_ci", True, "C(t)-index", "C(t)-index")
        if metric == "ci" else
        ("val_td_bs", "test_td_bs", False, "Brier(t)", "Brier score"))
    sel = pick(land, ["algorithm", "dataset", "seed", "landmark", "horizon"],
               val_col, test_col, mx)
    lms = {d: sorted(sel[(sel.dataset == d) & (sel.landmark > 0)].landmark.unique())
           for d in DS}
    nrow = max(len(v) for v in lms.values())
    fig, axes = plt.subplots(nrow, len(DS), figsize=(4.0 * len(DS), 2.5 * nrow),
                             squeeze=False)
    for col, d in enumerate(DS):
        for row in range(nrow):
            ax = axes[row][col]
            if row >= len(lms[d]):
                ax.axis("off")
                continue
            lm = lms[d][row]
            sub = sel[(sel.dataset == d) & (sel.landmark == lm)]
            hz = sorted(sub.horizon.unique())
            subs = [sub[sub.horizon == h] for h in hz]
            m, s, sig = agg(subs, algos, test_col, higher_better=mx)
            # landmark goes in the TITLE, not inside the axes: an in-axes
            # annotation collided with the significance asterisks.
            ttl = (f"{LAB[d]}\nlandmark t={lm}" if row == 0
                   else f"landmark t={lm}")
            grouped_bars(ax, hz, m, s, sig,
                         ylab if col == 0 else "", ttl,
                         [f"\u0394={h}" for h in hz])
            if metric == "ci":
                ax.set_ylim(0.4, 1.08)
            ax.title.set_fontsize(9 if row else 10)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, [f"{family} {a}" for a in l], loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.012))
    fig.text(0.985, -0.005,
             "* p<0.05   ** p<0.01   *** p<0.001   \u2020 D-TCSR worse (p<0.05)",
             fontsize=7.5, ha="right", color="0.35")
    fig.suptitle(f"{family} \u2014 time-dependent {what} by landmark and horizon",
                 fontsize=11, y=1.005)
    fig.tight_layout(rect=[0, 0.035, 1, 0.975])
    for ext in ("pdf", "png"):
        fig.savefig(out.with_suffix("." + ext), bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/final", type=Path)
    ap.add_argument("--out", default="results/final", type=Path)
    a = ap.parse_args()
    flat = pd.read_csv(a.results / "all_runs.csv")
    land = pd.read_csv(a.results / "all_landmarks.csv")
    fig_tcsr(flat, "LH", a.out / "summary_LH.pdf")
    fig_tcsr(flat, "DDH", a.out / "summary_DDH_tcsr_protocol.pdf")
    fig_landmarks(land, "DDH", a.out / "summary_DDH.pdf", metric="ci")
    fig_landmarks(land, "DDH", a.out / "summary_DDH_brier.pdf", metric="bs")
    fig_landmarks(land, "LH", a.out / "summary_LH_landmarks.pdf", metric="ci")
    fig_landmarks(land, "LH", a.out / "summary_LH_landmarks_brier.pdf", metric="bs")


if __name__ == "__main__":
    main()
