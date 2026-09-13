"""Hydra entrypoint: train + evaluate one DeepTCSR run."""

from __future__ import annotations

import os

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from survan.data import BatchIterator, load_dataset, train_val_test_split
from survan.model import DeepTCSR, DeepTCSRConfig


def _make_split_gen(split: dict, batch_size: int, shuffle: bool):
    return BatchIterator(
        {k: v for k, v in split.items() if v is not None},
        batch_size=batch_size, shuffle=shuffle,
    )


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    done = os.path.join(cfg.run_dir, "results_test.json")
    if os.path.exists(done):
        print(f"skip (already done): {cfg.run_dir}")
        return
    print(OmegaConf.to_yaml(cfg))
    ds = cfg.dataset
    seqs, ts, cs, target, h_ws, mask = load_dataset(
        ds.name,
        landmark=ds.landmark,
        compute_targets=True,
        kwargs={"data_path": ds.data_path, "horizon": ds.horizon},
    )
    arrays = {"X": seqs.astype(np.float32),
              "target": target.astype(np.float32),
              "h_ws": h_ws.astype(np.float32),
              "mask": mask.astype(np.float32)}
    if cfg.get("test_seed") is not None:
        # Learning-curve protocol: test set fixed by test_seed across all
        # seeds/sizes; the run seed reshuffles the remainder into train/val;
        # n_train truncates the (shuffled) train pool.
        s1 = train_val_test_split(
            arrays, ts, cs, seed=cfg.test_seed, test_size=cfg.test_size,
            val_size=None, stratify=ds.stratify)
        rest = s1["train"]
        arrays2 = {k: rest[k] for k in arrays}
        val_frac = cfg.val_size / (1.0 - cfg.test_size)
        s2 = train_val_test_split(
            arrays2, rest["ts"], rest["cs"], seed=cfg.seed,
            test_size=val_frac, val_size=None, stratify=ds.stratify)
        splits = {"train": s2["train"], "val": s2["test"], "test": s1["test"]}
        if cfg.get("n_train") is not None:
            n = int(cfg.n_train)
            avail = len(splits["train"]["ts"])
            if n > avail:
                raise ValueError(f"n_train={n} > available train pool {avail}")
            splits["train"] = {k: v[:n] for k, v in splits["train"].items()}
    else:
        splits = train_val_test_split(
            arrays, ts, cs,
            seed=cfg.seed, test_size=cfg.test_size, val_size=cfg.val_size,
            stratify=ds.stratify,
        )

    train_gen = _make_split_gen(splits["train"], ds.batch_size, shuffle=True)
    val_gen = _make_split_gen(splits["val"], ds.batch_size, shuffle=False)
    test_split = splits["test"]

    model_cfg = DeepTCSRConfig(
        horizon=ds.horizon,
        feature_dim=seqs.shape[-1],
        backbone=cfg.backbone.name,
        backbone_kwargs=OmegaConf.to_container(cfg.backbone.kwargs, resolve=True),
        learning_rate=ds.learning_rate,
        weight_decay=ds.weight_decay,
        num_epochs=ds.num_epochs,
        batch_size=ds.batch_size,
        lambda_=cfg.algorithm.lambda_,
        target_lr=cfg.algorithm.target_lr,
        tc=cfg.algorithm.get("tc", None),
        ranking_weight=cfg.algorithm.ranking_weight,
        ranking_sigma=cfg.algorithm.get("ranking_sigma", 1.0),
        cov_pred_weight=cfg.algorithm.cov_pred_weight,
        loss_norm=cfg.algorithm.loss_norm,
        weight_by_h_ws=cfg.algorithm.get("weight_by_h_ws", True),
        seed=cfg.seed,
        verbose=True,
    )

    model = DeepTCSR(model_cfg, sample_x=splits["train"]["X"])
    model.train(train_gen, val_gen=val_gen)

    tr_ts, tr_cs = splits["train"]["ts"], splits["train"]["cs"]
    test_ci, test_ci_ipcw, test_ibs, test_ibs_ipcw = model.evaluate(
        test_split["X"], test_split["ts"], test_split["cs"], tr_ts, tr_cs)
    val_ci, val_ci_ipcw, val_ibs, val_ibs_ipcw = model.evaluate(
        splits["val"]["X"], splits["val"]["ts"], splits["val"]["cs"],
        tr_ts, tr_cs)

    out = cfg.run_dir
    os.makedirs(out, exist_ok=True)
    model.save_results(out, {"split": "test", "ci": test_ci, "bs": test_ibs,
                             "ci_ipcw": test_ci_ipcw, "bs_ipcw": test_ibs_ipcw})
    os.rename(os.path.join(out, "results.json"),
              os.path.join(out, "results_test.json"))
    model.save_results(out, {"split": "val", "ci": val_ci, "bs": val_ibs,
                             "ci_ipcw": val_ci_ipcw, "bs_ipcw": val_ibs_ipcw})
    os.rename(os.path.join(out, "results.json"),
              os.path.join(out, "results_val.json"))
    model.save(out)
    print(f"test ci={test_ci:.4f} ({test_ci_ipcw:.4f} ipcw)  "
          f"bs={test_ibs:.4f} ({test_ibs_ipcw:.4f} ipcw)  → {out}")


if __name__ == "__main__":
    main()
