"""Summarize the architecture pilot and record the selected backbones.

Reads ``outputs_arch/<algorithm>/<dataset>/<backbone-label>/...`` and ranks
configurations by validation metric, averaged over seeds and over the two
pilot datasets (nasa, scania). Selection is on VALIDATION only; test values
are printed alongside purely for the record.

One winner is chosen per family (transformer for Cox, gru_attn for DDH) and
written to ``results/arch_pilot/selected_arch.json`` + a markdown table, so
the choice and the numbers behind it are documented for the paper.

Usage: uv run python scripts/arch_pilot_table.py [--root outputs_arch]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

FAMILY = {"tc_cox": "transformer (Cox family)", "tc_ddh": "gru_attn (DDH family)"}


def collect(root: Path) -> pd.DataFrame:
    rows = []
    for f in root.rglob("results_val.json"):
        t = f.parent / "results_test.json"
        if not t.exists():
            continue
        p = f.parts
        algo, ds, arch = p[1], p[2], p[3]
        seed = int(p[7].split("_")[1])
        v, te = json.load(open(f)), json.load(open(t))
        rows.append({"algorithm": algo, "dataset": ds, "arch": arch, "seed": seed,
                     "val_ci": v["ci"], "val_bs": v["bs"],
                     "test_ci": te["ci"], "test_bs": te["bs"]})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs_arch", type=Path)
    ap.add_argument("--out", default="results/arch_pilot", type=Path)
    args = ap.parse_args()

    df = collect(args.root)
    if df.empty:
        raise SystemExit(f"no pilot results under {args.root}")
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "all_runs.csv", index=False)

    selected, md = {}, ["# Architecture pilot\n",
                        "Selection on validation C-index (mean over the two pilot "
                        "datasets, 2 seeds each), tie-broken by validation IBS.\n"]
    for algo, label in FAMILY.items():
        sub = df[df.algorithm == algo]
        if sub.empty:
            continue
        # average per dataset first so neither dataset dominates by scale
        per_ds = sub.groupby(["arch", "dataset"])[
            ["val_ci", "val_bs", "test_ci", "test_bs"]].mean().reset_index()
        agg = per_ds.groupby("arch")[
            ["val_ci", "val_bs", "test_ci", "test_bs"]].mean().reset_index()
        agg = agg.sort_values(["val_ci", "val_bs"], ascending=[False, True])
        best = agg.iloc[0]
        selected[algo] = {"backbone": best["arch"],
                          "val_ci": round(float(best.val_ci), 4),
                          "val_bs": round(float(best.val_bs), 4),
                          "test_ci": round(float(best.test_ci), 4),
                          "test_bs": round(float(best.test_bs), 4)}
        print(f"\n=== {label} ===")
        print(agg.round(4).to_string(index=False))
        print(f"selected: {best['arch']}")
        md += [f"\n## {label}\n", agg.round(4).to_markdown(index=False),
               f"\n\n**Selected: `{best['arch']}`** "
               f"(val CI {best.val_ci:.4f}, val IBS {best.val_bs:.4f})\n",
               "\nPer-dataset breakdown:\n",
               per_ds.round(4).to_markdown(index=False), "\n"]

    (args.out / "selected_arch.json").write_text(json.dumps(selected, indent=2))
    (args.out / "arch_pilot.md").write_text("\n".join(md))
    print(f"\nWrote {args.out}/selected_arch.json and arch_pilot.md")


if __name__ == "__main__":
    main()
