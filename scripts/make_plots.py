"""Reproduce summary_COX.pdf and summary_DDH.pdf from outputs/.

Each plot is a grouped bar chart over four datasets with paired t-test
significance stars. Per (algorithm, dataset, seed) we select the
hyper-parameter combo with the best validation C-Index, then average the
corresponding test metric across seeds.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from aggregate import best_val_per_seed, collect


DATASET_LABEL = {
    "nasa": "NASA",
    "big_rw": "Large-RW",
    "mimic": "MIMIC",
    "churn_lastfm_months": "LastFM",
}


def _stars(p):
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return ""


def _paired(a, b):
    if len(a) < 2 or len(b) < 2 or len(a) != len(b):
        return ""
    try:
        return _stars(stats.ttest_rel(a, b, nan_policy="omit").pvalue)
    except Exception:
        return ""


def _summarize(df, baseline_algo, treatment_algo, datasets, metric):
    """Return dict[dataset] -> (mean_b, sem_b, mean_t, sem_t, stars)."""
    out = {}
    for ds in datasets:
        sub_b = df[(df.algorithm == baseline_algo) & (df.dataset == ds)]
        sub_t = df[(df.algorithm == treatment_algo) & (df.dataset == ds)]
        b = sub_b.set_index("seed")[metric].sort_index()
        t = sub_t.set_index("seed")[metric].sort_index()
        common = b.index.intersection(t.index)
        b, t = b.loc[common].values, t.loc[common].values
        if len(b) == 0:
            out[ds] = (np.nan, np.nan, np.nan, np.nan, "")
            continue
        out[ds] = (np.nanmean(b), stats.sem(b, nan_policy="omit"),
                   np.nanmean(t), stats.sem(t, nan_policy="omit"),
                   _paired(b, t))
    return out


def _bar(ax, summary, datasets, baseline_label, treatment_label, ylabel, title):
    n = len(datasets)
    x = np.arange(n)
    w = 0.38
    b_means = [summary[d][0] for d in datasets]
    b_errs = [summary[d][1] for d in datasets]
    t_means = [summary[d][2] for d in datasets]
    t_errs = [summary[d][3] for d in datasets]
    stars = [summary[d][4] for d in datasets]

    ax.bar(x - w / 2, b_means, w, yerr=b_errs, capsize=4,
           color="#4472C4", label=baseline_label)
    ax.bar(x + w / 2, t_means, w, yerr=t_errs, capsize=4,
           color="#D87A7A", hatch="//", edgecolor="white", label=treatment_label)

    for i, (tm, te, s) in enumerate(zip(t_means, t_errs, stars)):
        if s and not np.isnan(tm):
            ax.text(i + w / 2, tm + (te if not np.isnan(te) else 0) + 0.01,
                    s, ha="center", color="#C0392B", fontsize=12)

    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABEL.get(d, d) for d in datasets])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.4)


def _figure(df, baseline, treatment, datasets, baseline_label, treatment_label,
            suptitle, ibs_label, out_path):
    df_sel = best_val_per_seed(df, metric="val_ci", maximize=True)
    df_sel = df_sel[df_sel.algorithm.isin([baseline, treatment])
                    & df_sel.dataset.isin(datasets)]
    ci_summary = _summarize(df_sel, baseline, treatment, datasets, "test_ci")
    bs_summary = _summarize(df_sel, baseline, treatment, datasets, "test_bs")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    _bar(axes[0], ci_summary, datasets, baseline_label, treatment_label,
         "C-Index", "C-Index  (↑ higher is better)")
    _bar(axes[1], bs_summary, datasets, baseline_label, treatment_label,
         ibs_label, f"{ibs_label}  (↓ lower is better)")
    fig.suptitle(suptitle, fontweight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.08, -0.02),
               ncol=2, frameon=False)
    fig.text(0.62, -0.02, "* p<0.05    ** p<0.01    *** p<0.001", fontsize=9)
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs", type=Path)
    ap.add_argument("--out", default=".", type=Path)
    args = ap.parse_args()
    df = collect(args.root)
    if df.empty:
        raise SystemExit(f"No runs found under {args.root}")

    cox_datasets = ["nasa", "big_rw", "mimic", "churn_lastfm_months"]
    _figure(df, "cox", "tc_cox", cox_datasets,
            "Cox (best landmark)", "TC-Cox",
            "Cox vs TC-Cox — Test performance (best-val config)",
            "Brier Score", args.out / "summary_COX.pdf")

    ddh_datasets = ["nasa", "big_rw", "mimic", "churn_lastfm_months"]
    _figure(df, "ddh", "tc_ddh", ddh_datasets,
            "DDH", "TC-DDH",
            "DDH vs TC-DDH — Test performance (best-val config)",
            "IBS", args.out / "summary_DDH.pdf")


if __name__ == "__main__":
    main()
