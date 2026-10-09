"""Final large-benchmark tables: baseline vs Inc-TCSR vs D-TCSR.

Reads outputs/ and reports both evaluation protocols:

  * TCSR protocol  -- read-out at state 0, global C-index / IBS. Faithful to
    Maystre & Russo, where "landmark" is a TRAINING scheme and all arms are
    scored from the initial state.
  * Dynamic-DeepHit protocol -- landmark x horizon grid, conditional risk
    F(t_M + delta | T > t_M), time-dependent C(t)-index and Brier score.
    Faithful to Lee et al.; landmark 0 is the TCSR read-out point.

Hyper-parameters (lambda, and tau for D-TCSR) are selected on VALIDATION,
separately per reported metric, never on test.
"""
from __future__ import annotations
import json, os
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

DS = ["nasa", "scania", "churn_lastfm_months", "big_rw"]
LABEL = {"nasa": "NASA", "scania": "Scania",
         "churn_lastfm_months": "LastFM", "big_rw": "Large-RW"}
FAM = {"LH": ["cox", "inc_tc_cox", "tc_cox"],
       "DDH": ["ddh", "inc_tc_ddh", "tc_ddh"]}
NAME = {"cox": "baseline", "inc_tc_cox": "Inc-TCSR", "tc_cox": "D-TCSR",
        "ddh": "baseline", "inc_tc_ddh": "Inc-TCSR", "tc_ddh": "D-TCSR"}


def load(root=Path(os.environ.get("OUTPUT_DIR", "outputs"))):
    flat, land = [], []
    for f in root.rglob("results_test.json"):
        p = f.parts
        algo, ds = p[1], p[2]
        lam = float(p[4].split("_")[1]); tau = float(p[5].split("_")[2])
        seed = int(p[7].split("_")[1])
        te = json.load(open(f)); va = json.load(open(f.parent / "results_val.json"))
        key = dict(algorithm=algo, dataset=ds, lambda_=lam, target_lr=tau, seed=seed)
        flat.append({**key, "val_ci": va["ci"], "val_bs": va["bs"],
                     "test_ci": te["ci"], "test_bs": te["bs"]})
        lt = {(r["landmark"], r["horizon"]): r
              for r in json.load(open(f.parent / "landmarks_fixed_test.json"))}
        lv = {(r["landmark"], r["horizon"]): r
              for r in json.load(open(f.parent / "landmarks_fixed_val.json"))}
        for k in lt.keys() & lv.keys():
            land.append({**key, "landmark": k[0], "horizon": k[1],
                         "n_at_risk": lt[k]["n_at_risk"],
                         "val_td_ci": lv[k]["td_ci"], "test_td_ci": lt[k]["td_ci"],
                         "val_td_bs_ipcw": lv[k]["td_bs_ipcw"],
                         "test_td_bs_ipcw": lt[k]["td_bs_ipcw"]})
    return pd.DataFrame(flat), pd.DataFrame(land)


from selection import pick_global


def pick(df, keys, val_col, test_col, maximize):
    """One config per (algorithm, dataset[, ...]) on MEAN validation across
    seeds -- not a different winner per seed. `keys` may still contain
    "seed"; it is dropped, since seeds are what we average over."""
    keys = [k for k in keys if k != "seed"]
    return pick_global(df, keys, val_col, test_col, maximize)


def cell(g, col):
    return (f"{g[col].mean():.3f}±{stats.sem(g[col]):.3f}"
            if len(g) > 1 else ("—" if g.empty else f"{g[col].mean():.3f}"))


def main():
    flat, land = load()
    out = Path("results/final"); out.mkdir(parents=True, exist_ok=True)
    flat.to_csv(out / "all_runs.csv", index=False)
    land.to_csv(out / "all_landmarks.csv", index=False)

    for metric, val_col, maximize, title in (
            ("test_ci", "val_ci", True, "Test C-index (sel. val C-index)"),
            ("test_bs", "val_bs", False, "Test IBS (sel. val IBS)")):
        sel = pick(flat, ["algorithm", "dataset", "seed"], val_col, metric, maximize)
        for fam, algos in FAM.items():
            print(f"\n=== {fam} family — {title} [TCSR protocol, state 0] ===")
            rows = []
            for a in algos:
                r = {"arm": NAME[a]}
                for ds in DS:
                    r[LABEL[ds]] = cell(sel[(sel.algorithm == a) & (sel.dataset == ds)], metric)
                rows.append(r)
            print(pd.DataFrame(rows).to_string(index=False))
            # D-TCSR is tested against BOTH references: beating Inc-TCSR
            # means little if the untreated baseline is better still.
            base, inc, dt = algos
            for ds in DS:
                bas = sel[(sel.algorithm == base) & (sel.dataset == ds)].set_index("seed")[metric]
                a = sel[(sel.algorithm == inc) & (sel.dataset == ds)].set_index("seed")[metric]
                b = sel[(sel.algorithm == dt) & (sel.dataset == ds)].set_index("seed")[metric]
                c = a.index.intersection(b.index)
                cb = bas.index.intersection(b.index)
                if len(c) > 1 and len(cb) > 1:
                    p1 = stats.ttest_rel(b.loc[c], a.loc[c]).pvalue
                    p2 = stats.ttest_rel(b.loc[cb], bas.loc[cb]).pvalue
                    print(f"   {LABEL[ds]:9s} D-TCSR vs Inc = {(b.loc[c]-a.loc[c]).mean():+.4f} p={p1:.4f}"
                          f"   |  vs baseline = {(b.loc[cb]-bas.loc[cb]).mean():+.4f} p={p2:.4f}")

    # --- DDH protocol: landmark x horizon, time-dependent C-index ---
    # One config per (algorithm, dataset) on mean val over landmark > 0 cells:
    # the same rule as select_configs.py (which chose what got 30 seeds).
    sel = pick(land[land.landmark > 0], ["algorithm", "dataset", "seed"],
               "val_td_ci", "test_td_ci", True)
    for ds in DS:
        sub = sel[sel.dataset == ds]
        if sub.empty:
            continue
        print(f"\n=== {LABEL[ds]} — time-dependent C(t)-index [DDH protocol] ===")
        rows = []
        for fam, algos in FAM.items():
            for a in algos:
                r = {"family": fam, "arm": NAME[a]}
                for (lm, hz), g in sorted(sub[sub.algorithm == a].groupby(["landmark", "horizon"])):
                    r[f"t={lm},d={hz}"] = cell(g, "test_td_ci")
                rows.append(r)
        print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nWrote {out}/all_runs.csv and all_landmarks.csv")


if __name__ == "__main__":
    main()
