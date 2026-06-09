"""Unified DeepTCSR model — covers Cox, TC-Cox, DDH, TC-DDH."""

from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Mapping

import chex
import jax
import jax.numpy as jnp
import numpy as np
import optax

from .backbones import build_backbone
from .data import BatchIterator, get_targets_and_masks, to_jax
from .losses import (covariate_prediction_loss, hazard_bce, median_survival_time,
                     ranking_loss, survival_curve, target_survival_weights,
                     tc_targets, tc_weights)
from .metrics import concordance_index, integrated_brier_score_np


@chex.dataclass(frozen=True)
class _State:
    params: Any
    tgt_params: Any
    opt_state: optax.OptState


@dataclass
class DeepTCSRConfig:
    """Configuration for DeepTCSR.

    Aux loss weights default to 0 (pure hazard). lambda_=0 + target_lr=1.0
    recovers the plain online network without temporal consistency.
    """
    horizon: int
    feature_dim: int
    backbone: str
    backbone_kwargs: Mapping[str, Any] = field(default_factory=dict)
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    lr_decay_rate: float = 1.0
    warmup_epochs: int = 0
    lambda_: float = 0.0
    target_lr: float = 1.0
    ranking_weight: float = 0.0
    ranking_sigma: float = 1.0
    cov_pred_weight: float = 0.0
    loss_norm: str = "mean"
    axis: int = 2
    num_epochs: int = 100
    batch_size: int = 64
    early_stopping_patience: int = 5
    log_interval: int = 10
    verbose: bool = False
    seed: int = 0


class DeepTCSR:
    """Single class for hazard models with optional target-net soft TC targets
    and optional ranking / covariate-prediction auxiliary losses."""

    def __init__(self, cfg: DeepTCSRConfig, sample_x):
        self.cfg = cfg
        self.horizon = cfg.horizon
        self._key = jax.random.PRNGKey(cfg.seed)

        self.backbone = build_backbone(
            cfg.backbone, horizon=cfg.horizon,
            feature_dim=cfg.feature_dim, seed=cfg.seed,
            **cfg.backbone_kwargs,
        )

        params = self.backbone.init(self._next_key(), jnp.asarray(sample_x[:1]))
        tgt_params = jax.tree_util.tree_map(jnp.copy, params)
        self._lr_schedule = self._make_schedule()
        opt = optax.adamw(self._lr_schedule, weight_decay=cfg.weight_decay)
        self._opt = opt
        self.state = _State(params=params, tgt_params=tgt_params,
                            opt_state=opt.init(params))

        self._tgt_update = partial(optax.incremental_update,
                                   step_size=cfg.target_lr)
        self._update = jax.jit(self._build_update())

    def _next_key(self):
        self._key, sub = jax.random.split(self._key)
        return sub

    def _make_schedule(self):
        if self.cfg.warmup_epochs <= 0 or self.cfg.lr_decay_rate == 1.0:
            return optax.constant_schedule(self.cfg.learning_rate)
        return optax.join_schedules(
            schedules=[
                optax.constant_schedule(self.cfg.learning_rate),
                optax.exponential_decay(
                    init_value=self.cfg.learning_rate,
                    transition_steps=1,
                    decay_rate=self.cfg.lr_decay_rate,
                ),
            ],
            boundaries=[self.cfg.warmup_epochs],
        )

    # ----- loss / update -----

    def _compute_loss(self, params, x, targets, mask, weights, ts, cs):
        logits, aux = self.backbone.apply(params, x)
        h_loss = hazard_bce(logits, targets, mask, weights=weights,
                            mode=self.cfg.loss_norm)
        total = h_loss
        if self.cfg.ranking_weight > 0:
            total = total + self.cfg.ranking_weight * ranking_loss(
                logits, ts, cs, axis=self.cfg.axis, sigma=self.cfg.ranking_sigma)
        if self.cfg.cov_pred_weight > 0 and aux is not None:
            total = total + self.cfg.cov_pred_weight * covariate_prediction_loss(
                aux, x)
        return total

    def _build_update(self):
        grad_fn = jax.value_and_grad(self._compute_loss)

        def update(state: _State, x, targets, mask, weights, ts, cs):
            loss, grads = grad_fn(state.params, x, targets, mask, weights, ts, cs)
            updates, opt_state = self._opt.update(grads, state.opt_state, state.params)
            params = optax.apply_updates(state.params, updates)
            tgt_params = self._tgt_update(params, state.tgt_params)
            return state.replace(params=params, tgt_params=tgt_params,
                                 opt_state=opt_state), loss
        return update

    # ----- target builders -----

    def _hard_targets(self, x, ts, cs):
        x_np = np.asarray(x)
        ts_np = np.asarray(ts)
        cs_np = np.asarray(cs)
        landmark = self.cfg.backbone == "linear"  # legacy convention
        y, h_ws, mask = get_targets_and_masks(x_np, ts_np, cs_np, landmark)
        return to_jax(y.astype(np.float32),
                      h_ws.astype(np.float32),
                      mask.astype(np.float32))

    def _maybe_soft_targets(self, x, hard_y, h_ws, cs):
        """Blend hard targets with the target network forecast when lambda_>0."""
        if self.cfg.lambda_ <= 0.0:
            return hard_y, h_ws
        tgt_logits, _ = self.backbone.apply(self.state.tgt_params, x)
        s_ws = target_survival_weights(tgt_logits)
        soft_y = tc_targets(jax.nn.sigmoid(tgt_logits), hard_y,
                            self.cfg.lambda_, self.cfg.horizon)
        soft_w = tc_weights(s_ws, h_ws, hard_y, cs,
                            self.cfg.lambda_, self.cfg.horizon)
        return soft_y, soft_w

    # ----- training -----

    def train(self, train_gen: BatchIterator, val_gen: BatchIterator | None = None):
        best_val = float("inf")
        patience = 0
        history = []
        for epoch in range(self.cfg.num_epochs):
            losses = []
            for batch in train_gen:
                x, ts, cs = (jnp.asarray(batch["X"]),
                             jnp.asarray(batch["ts"]),
                             jnp.asarray(batch["cs"]))
                if "target" in batch:
                    y, h_ws, mask = (jnp.asarray(batch["target"]).astype(jnp.float32),
                                     jnp.asarray(batch["h_ws"]).astype(jnp.float32),
                                     jnp.asarray(batch["mask"]).astype(jnp.float32))
                else:
                    y, h_ws, mask = self._hard_targets(x, ts, cs)
                soft_y, soft_w = self._maybe_soft_targets(x, y, h_ws, cs)
                self.state, loss = self._update(
                    self.state, x, soft_y, mask, soft_w, ts, cs)
                losses.append(float(loss))
            train_gen.reset()
            mean_loss = float(np.mean(losses))
            history.append(mean_loss)

            if val_gen is not None:
                vl = self._eval_loss(val_gen)
                if vl < best_val - 1e-4:
                    best_val, patience = vl, 0
                else:
                    patience += 1
                if patience >= self.cfg.early_stopping_patience:
                    if self.cfg.verbose:
                        print(f"Early stop at epoch {epoch}")
                    break
                if self.cfg.verbose and epoch % self.cfg.log_interval == 0:
                    print(f"epoch {epoch:03d}  train {mean_loss:.4f}  "
                          f"val {vl:.4f}  best {best_val:.4f}")
            elif self.cfg.verbose and epoch % self.cfg.log_interval == 0:
                print(f"epoch {epoch:03d}  train {mean_loss:.4f}")
        return history

    def _eval_loss(self, gen):
        losses = []
        for batch in gen:
            x, ts, cs = (jnp.asarray(batch["X"]),
                         jnp.asarray(batch["ts"]),
                         jnp.asarray(batch["cs"]))
            if "target" in batch:
                y, h_ws, mask = (jnp.asarray(batch["target"]).astype(jnp.float32),
                                 jnp.asarray(batch["h_ws"]).astype(jnp.float32),
                                 jnp.asarray(batch["mask"]).astype(jnp.float32))
            else:
                y, h_ws, mask = self._hard_targets(x, ts, cs)
            soft_y, soft_w = self._maybe_soft_targets(x, y, h_ws, cs)
            loss = self._compute_loss(self.state.params, x, soft_y, mask,
                                      soft_w, ts, cs)
            losses.append(float(loss))
        gen.reset()
        return float(np.mean(losses))

    # ----- evaluation -----

    def survival_curve(self, x):
        logits, _ = self.backbone.apply(self.state.params, jnp.asarray(x))
        return survival_curve(logits, axis=self.cfg.axis)

    def evaluate(self, x, ts, cs):
        """Return (ci, ibs) on a single batch (numpy arrays)."""
        surv = self.survival_curve(x)
        scores = np.asarray(median_survival_time(surv))
        ci = float(concordance_index(scores, ts, cs))
        ibs = integrated_brier_score_np(
            np.asarray(surv[:, 1:]), np.asarray(ts), np.asarray(cs))
        return ci, ibs

    # ----- I/O -----

    def save(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "model.pkl"), "wb") as f:
            pickle.dump({"params": self.state.params,
                         "tgt_params": self.state.tgt_params}, f)

    def save_results(self, output_dir, results: Mapping[str, Any]):
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "results.json"), "w") as f:
            json.dump({k: (float(v) if isinstance(v, (np.floating, jnp.ndarray)) else v)
                       for k, v in results.items()}, f)
