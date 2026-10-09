"""Stabilizing effect of the target network: tau ablation on random walks.

Experiment of the paper's "Stabilizing effect of the target network"
subsection:

  * 20-dim Gauss-Markov random walks (tdsurv generator), horizons
    H in {30, 50, 100}; the churn bias is calibrated per H for ~20% censoring.
  * For each (H, tau): 30 independent training (and validation) sets, one
    shared test set of 1000 sequences.
  * LH model with the LargeRW setup of the main experiments (transformer
    64x2, AdamW lr 1e-2, wd 1e-4, batch 128), temporal consistency with a
    fixed lambda, target network updated after every step with rate tau; the
    model initialization is fixed (seed 42) so the spread across runs comes
    from the training data, as in the original.

  Protocol (environment variables; defaults = the paper's setting):
    ABL_OUT       output dir                  results/tau_ablation_main
    ABL_LAMBDA    TC lambda                   0.1  (LargeRW D-TCSR selection)
    ABL_NTRAIN    training sequences          1000 (~8 steps per epoch)
    ABL_NVAL      validation sequences        500  (0 = no early stopping)
    ABL_PATIENCE  early-stopping patience     5 epochs
    ABL_EPOCHS    max epochs                  1000
  * Metrics on the test set (state-0 protocol): C-index and IBS.
  * Variability of the estimates: for every valid test entry (subject i,
    state l, horizon k), mean and std of h_theta(k | x_l) across the 30 runs;
    delta = std / sqrt(h(1-h)) (used in the paper; figures *_sqrt.png) and
    std / (h(1-h)) (figures *_paper.png).

    uv run python scripts/tau_ablation.py run --horizon 30 --tau 0.1 --run 0
    uv run python scripts/tau_ablation.py aggregate   # metrics.csv, table, figures
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.special import expit as sigmoid

env = os.environ.get
OUT = Path(env("ABL_OUT", "results/tau_ablation_main"))
LAMBDA = float(env("ABL_LAMBDA", "0.1"))
N_TRAIN, N_VAL = int(env("ABL_NTRAIN", "1000")), int(env("ABL_NVAL", "500"))
PATIENCE, EPOCHS = int(env("ABL_PATIENCE", "5")), int(env("ABL_EPOCHS", "1000"))
HORIZONS = (30, 50, 100)
BIAS = {30: -2.5, 50: -3.5, 100: -5.0}   # ~20% censoring (calibrated)
TAUS = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0)
N_RUNS, N_TEST, D, MODEL_SEED = 30, 1000, 20, 42


def generate(n, horizon, rng):
    """tdsurv random walk: x0 ~ N(0, 1), x' = x + N(0, .5^2), churn w.p.
    sigmoid(x . theta + bias). Same layout as prepare_small_data.prepare_rw:
    seqs (n, horizon+1, D+1) with a constant last feature; ts = number of
    states before the event, censored at ts = horizon."""
    thetas = np.random.default_rng(0).normal(size=D)
    seqs = np.zeros((n, horizon + 1, D + 1), dtype=np.float32)
    ts = np.ones(n, dtype=int)
    cs = np.zeros(n, dtype=bool)
    for i in range(n):
        x = rng.normal(size=D)
        seqs[i, 0] = np.append(x, 1)
        for _ in range(horizon):
            if rng.uniform() < sigmoid(x @ thetas + BIAS[horizon]):
                break
            x = x + 0.5 * rng.normal(size=D)
            seqs[i, ts[i]] = np.append(x, 1)
            ts[i] += 1
        if ts[i] > horizon:
            ts[i], cs[i] = horizon, True
    return seqs, ts, cs


def test_set(horizon):
    return generate(N_TEST, horizon, np.random.default_rng([horizon, 10**6]))


def run_one(horizon, tau, run):
    out = OUT / f"H{horizon}" / f"tau_{tau}" / f"run_{run}.npz"
    if out.exists():
        return
    import jax
    import jax.numpy as jnp
    from survan.data import BatchIterator, get_targets_and_masks
    from survan.model import DeepTCSR, DeepTCSRConfig

    H = horizon + 1
    x, ts, cs = generate(N_TRAIN, horizon, np.random.default_rng([horizon, run]))
    # landmark targets precomputed, as run.py does (the model only builds
    # landmark targets itself for the linear backbone)
    y, h_ws, mask = get_targets_and_masks(x, ts, cs, landmark=True)
    np.random.seed(run)  # BatchIterator shuffles with the global RNG
    gen = BatchIterator({"X": x, "ts": ts, "cs": cs, "target": y.astype(np.float32),
                         "h_ws": h_ws, "mask": mask}, batch_size=128, shuffle=True)
    val = None
    if N_VAL > 0:
        xv, tv, cv = generate(N_VAL, horizon, np.random.default_rng([horizon, run, 1]))
        yv, wv, mv = get_targets_and_masks(xv, tv, cv, landmark=True)
        val = BatchIterator({"X": xv, "ts": tv, "cs": cv, "target": yv.astype(np.float32),
                             "h_ws": wv, "mask": mv}, batch_size=128, shuffle=False)
    cfg = DeepTCSRConfig(
        horizon=H, feature_dim=D + 1, backbone="transformer",
        backbone_kwargs=dict(hidden_size=64, num_layers=2, seq_len=H, dropout=0.2),
        learning_rate=1e-2, weight_decay=1e-4, lambda_=LAMBDA, target_lr=tau, tc=True,
        loss_norm="weighted", num_epochs=EPOCHS, early_stopping_patience=PATIENCE,
        batch_size=128, seed=MODEL_SEED)
    model = DeepTCSR(cfg, sample_x=x)
    epochs = len(model.train(gen, val_gen=val))  # val None: no early stopping

    xt, tt, ct = test_set(horizon)
    ci, ci_ipcw, ibs, ibs_ipcw = model.evaluate(xt, tt, ct, ts, cs)
    _, ws_t, m_t = get_targets_and_masks(xt, tt, ct, landmark=True)
    valid = ws_t & m_t[:, :, :1]          # observed state l, known horizon k
    logits, _ = model.backbone.apply(model.state.params, jnp.asarray(xt))
    hz = np.asarray(jax.nn.sigmoid(logits))[valid]
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, hazards=hz.astype(np.float32), ci=ci, ci_ipcw=ci_ipcw,
                        ibs=ibs, ibs_ipcw=ibs_ipcw, epochs=epochs)


def aggregate():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from scipy import stats

    rows, delta_rows = [], []
    for horizon in HORIZONS:
        deltas = {}
        for tau in TAUS:
            files = sorted((OUT / f"H{horizon}" / f"tau_{tau}").glob("run_*.npz"))
            if not files:
                continue
            runs = [np.load(f) for f in files]
            for f, r in zip(files, runs):
                rows.append(dict(H=horizon, tau=tau, run=int(f.stem.split("_")[1]),
                                 ci=float(r["ci"]), ibs=float(r["ibs"]),
                                 ci_ipcw=float(r["ci_ipcw"]), ibs_ipcw=float(r["ibs_ipcw"]),
                                 epochs=int(r["epochs"]) if "epochs" in r.files else None))
            hz = np.stack([r["hazards"] for r in runs]).astype(np.float64)
            m, s = hz.mean(0), hz.std(0)
            ok = (m > 0) & (m < 1)
            deltas[tau] = {"paper": s[ok] / (m[ok] * (1 - m[ok])),
                           "sqrt": s[ok] / np.sqrt(m[ok] * (1 - m[ok]))}
            for kind, v in deltas[tau].items():
                delta_rows.append(dict(H=horizon, tau=tau, formula=kind, n_runs=len(runs),
                                       mean=v.mean(), median=np.median(v),
                                       p90=np.quantile(v, .9)))
        for kind, label in (("paper", r"$\delta = \sigma / (\bar h(1-\bar h))$"),
                            ("sqrt", r"$\delta = \sigma / \sqrt{\bar h(1-\bar h)}$")):
            if not deltas:
                continue
            fig, axs = plt.subplots(2, 3, figsize=(13, 6.5), sharex=True)
            hi = max(np.quantile(d[kind], .99) for d in deltas.values())
            for ax, (tau, d) in zip(axs.flat, deltas.items()):
                v = d[kind]
                ax.hist(v[v <= hi], bins=80, range=(0, hi), color="skyblue", density=True)
                ax.axvline(v.mean(), color="red")
                ax.set_title(rf"$\tau = {tau}$")
                ax.set_xlabel(f"variability {label}")
            fig.suptitle(f"H = {horizon}  (red: mean; x-axis clipped at the 99th percentile)")
            fig.tight_layout()
            fig.savefig(OUT / f"variability_tau_h{horizon}_{kind}.png", dpi=150)
            plt.close(fig)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "metrics.csv", index=False)
    pd.DataFrame(delta_rows).to_csv(OUT / "variability_summary.csv", index=False)

    # table: mean +/- std over runs; bold = best mean and every tau not
    # significantly different from it (paired t-test over runs, alpha=.05)
    lines = []
    for horizon in HORIZONS:
        for metric, higher, name in (("ci", True, r"CI$\uparrow$"), ("ibs", False, r"IBS$\downarrow$")):
            p = df[df.H == horizon].pivot(index="run", columns="tau", values=metric).dropna()
            if p.empty:
                continue
            means = p.mean()
            best = means.idxmax() if higher else means.idxmin()
            cells = []
            for tau in p.columns:
                txt = f"{means[tau]:.3f} $\\pm$ {p[tau].std(ddof=1):.3f}".replace("0.", ".")
                same = tau == best or stats.ttest_rel(p[best], p[tau]).pvalue >= 0.05
                cells.append(f"\\textbf{{{txt}}}" if same else txt)
            lead = f"\\multirow{{2}}{{*}}{{{horizon}}}" if metric == "ci" else ""
            lines.append(f"{lead} & {name} & " + " & ".join(cells) + r" \\")
        lines.append(r"\hline")
    (OUT / "table_ablation_tau.tex").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(pd.DataFrame(delta_rows).pivot_table(index=["H", "formula"], columns="tau",
                                               values="mean").round(3).to_string())


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--horizon", type=int, required=True)
    r.add_argument("--tau", type=float, required=True)
    r.add_argument("--run", type=int, required=True)
    sub.add_parser("aggregate")
    ls = sub.add_parser("list")  # one line per pending run, for xargs
    ls.add_argument("--horizons", type=int, nargs="*", default=list(HORIZONS))
    a = ap.parse_args()
    if a.cmd == "run":
        run_one(a.horizon, a.tau, a.run)
    elif a.cmd == "aggregate":
        aggregate()
    else:
        for horizon in a.horizons:
            for tau in TAUS:
                for run in range(N_RUNS):
                    if not (OUT / f"H{horizon}" / f"tau_{tau}" / f"run_{run}.npz").exists():
                        print(horizon, tau, run)


if __name__ == "__main__":
    main()
