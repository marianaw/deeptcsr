"""Recompute the DDH-protocol landmark grid on a FIXED per-dataset grid.

The sweep derived landmarks from each seed's own training split, so every
seed produced a different (landmark, horizon) grid and seeds could not be
pooled. Dynamic-DeepHit instead uses dataset-level constants (ages 30/40/50
in the reference), so the faithful analogue is one grid per dataset, shared
by every run. Recomputed offline from the saved model.pkl -- no retraining.

Writes landmarks_fixed_{test,val}.json next to each run's results.
"""
from __future__ import annotations
import json, pickle
from pathlib import Path
import numpy as np, yaml, jax, jax.numpy as jnp
from survan.data import load_dataset, train_val_test_split
from survan.backbones import build_backbone
from survan.metrics import td_concordance_index, td_brier_score

ROOT = Path("outputs_final")
DATA_ROOT = "../SurvanData"
ARCH = {"transformer": dict(hidden_size=64, num_layers=2),
        "gru_attn": dict(hidden_size=128, attention_hidden=16)}


def dataset_cfg(name):
    c = yaml.safe_load(open(f"configs/dataset/{name}.yaml"))
    path = c["data_path"].replace("${data_root}",
                                  "data" if name == "scania" else DATA_ROOT)
    return c, path


def fixed_grid(ts):
    """One grid per dataset: landmark 0 (TCSR read-out) + ts quartiles."""
    lms = [0] + [int(x) for x in np.percentile(ts, [25, 50, 75])]
    grid = {}
    for lm in sorted(set(lms)):
        rem = ts[ts > lm] - lm
        if len(rem) < 10:
            continue
        hs = sorted({int(h) for h in np.percentile(rem, [25, 50, 75]) if h >= 1})
        if hs:
            grid[lm] = hs
    return grid


def main():
    cache = {}
    runs = sorted(ROOT.rglob("model.pkl"),
                  key=lambda m: (m.parts[2], int(m.parts[7].split("_")[1])))
    # skip finished work so an interrupted pass resumes instead of restarting
    runs = [m for m in runs
            if not (m.parent / "landmarks_fixed_val.json").exists()]
    print(f"{len(runs)} runs to do")
    cur_split = (None, None, None)
    for i, mp in enumerate(runs, 1):
        p = mp.parts
        algo, ds, bb = p[1], p[2], p[3]
        seed = int(p[7].split("_")[1])
        if ds not in cache:
            cache.clear()  # one dataset resident at a time
            c, path = dataset_cfg(ds)
            seqs, ts, cs, *_ = load_dataset(ds, landmark=c["landmark"],
                                            compute_targets=False,
                                            kwargs={"data_path": path,
                                                    "horizon": c["horizon"]})
            cache[ds] = (c, seqs, ts, cs, fixed_grid(np.asarray(ts)), {})
        c, seqs, ts, cs, grid, _ = cache[ds]
        # only ONE split is held at a time: caching all 10 seeds kept ~10
        # copies of every dataset's arrays in memory and got the job killed.
        if cur_split[0] != ds or cur_split[1] != seed:
            cur_split = (ds, seed,
                         train_val_test_split({"X": seqs}, ts, cs, seed=seed,
                                              test_size=0.2, val_size=0.2,
                                              stratify=c["stratify"]))
        sp = cur_split[2]
        arch = "gru_attn" if "ddh" in algo else "transformer"
        kw = dict(ARCH[arch])
        if arch == "transformer":
            kw["seq_len"] = c["horizon"]
        model = build_backbone(arch, horizon=c["horizon"],
                               feature_dim=seqs.shape[-1], seed=seed, **kw)
        params = pickle.load(open(mp, "rb"))["params"]
        tr_ts, tr_cs = np.asarray(sp["train"]["ts"]), np.asarray(sp["train"]["cs"]).astype(bool)
        for split in ("test", "val"):
            x = sp[split]["X"]
            st = np.asarray(sp[split]["ts"]); sc = np.asarray(sp[split]["cs"]).astype(bool)
            logits, _ = model.apply(params, jnp.asarray(x))
            haz = np.asarray(jax.nn.sigmoid(logits))
            rows = []
            for lm, hs in grid.items():
                if lm >= haz.shape[1]:
                    continue
                ar = st > lm
                if ar.sum() < 10:
                    continue
                rem, rem_cs = (st - lm)[ar], sc[ar]
                h = np.clip(haz[ar, lm, :].astype(np.float64), 0.0, 1 - 1e-12)
                surv = np.exp(np.cumsum(np.log1p(-h), axis=1))
                tar = tr_ts > lm
                trm = (tr_ts - lm)[tar] if tar.sum() >= 10 else None
                trc = tr_cs[tar] if tar.sum() >= 10 else None
                for d in hs:
                    if d > surv.shape[1]:
                        continue
                    risk = 1.0 - surv[:, d - 1]
                    rows.append({"landmark": int(lm), "horizon": int(d),
                                 "n_at_risk": int(ar.sum()),
                                 "td_ci": td_concordance_index(risk, rem, rem_cs, d),
                                 "td_bs": td_brier_score(risk, rem, rem_cs, d),
                                 "td_ci_ipcw": td_concordance_index(risk, rem, rem_cs, d, trm, trc),
                                 "td_bs_ipcw": td_brier_score(risk, rem, rem_cs, d, trm, trc)})
            json.dump(rows, open(mp.parent / f"landmarks_fixed_{split}.json", "w"))
        if i % 50 == 0:
            print(f"  {i}/{len(runs)}", flush=True)
    for ds, (_, _, _, _, g, _) in cache.items():
        print(f"{ds}: grid {g}")


if __name__ == "__main__":
    main()
