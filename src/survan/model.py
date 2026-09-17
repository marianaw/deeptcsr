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
from .metrics import (concordance_index, concordance_index_ipcw,
                      td_brier_score, td_concordance_index,
                      integrated_brier_score_ipcw, integrated_brier_score_np)


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
    tc: bool | None = None  # None: derive from lambda_ > 0 (legacy behavior).
                            # Explicit True enables TC bootstrapping even at
                            # lambda_=0 (pure one-step bootstrap targets, as in
                            # the small-data TCSR protocol). target_lr=1.0 is
                            # then Inc-TCSR; target_lr<1 is D-TCSR.
    ranking_weight: float = 0.0
    ranking_sigma: float = 1.0
    cov_pred_weight: float = 0.0
    loss_norm: str = "mean"
    weight_by_h_ws: bool = True  # Cox/TC-Cox use m=h_ws (legacy convention);
                                 # plain DDH legacy uses mask only (set False).
                                 # TC paths always use blended soft_w (this flag
                                 # only affects the lambda_=0 branch).
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

        params = self.backbone.init(jax.random.PRNGKey(cfg.seed),
                                    jnp.asarray(sample_x[:1]))
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
        """Blend hard targets with the target network forecast when TC is on."""
        use_tc = self.cfg.tc if self.cfg.tc is not None else self.cfg.lambda_ > 0.0
        if not use_tc:
            return hard_y, (h_ws if self.cfg.weight_by_h_ws else None)
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
        """Validation loss = configured loss recipe against hard targets
        (no soft TC blending). For Cox/TC-Cox this collapses to masked BCE
        — matching legacy Cox `test_step`. For DDH/TC-DDH this also adds
        ranking + cov-prediction losses, matching legacy DDH `loss_fn`."""
        losses = []
        for batch in gen:
            x = jnp.asarray(batch["X"])
            ts = jnp.asarray(batch["ts"])
            cs = jnp.asarray(batch["cs"])
            if "target" in batch:
                y = jnp.asarray(batch["target"]).astype(jnp.float32)
                h_ws = jnp.asarray(batch["h_ws"]).astype(jnp.float32)
                mask = jnp.asarray(batch["mask"]).astype(jnp.float32)
            else:
                y, h_ws, mask = self._hard_targets(x, ts, cs)
            if self.cfg.loss_norm == "mean":
                # Legacy Cox val: BCE * mask only (no h_ws on val).
                logits, _ = self.backbone.apply(self.state.params, x)
                bce = optax.sigmoid_binary_cross_entropy(logits, y)
                total = jnp.mean(bce * mask)
            else:
                w_eval = h_ws if self.cfg.weight_by_h_ws else None
                total = self._compute_loss(self.state.params, x, y, mask,
                                           w_eval, ts, cs)
            losses.append(float(total))
        gen.reset()
        return float(np.mean(losses))

    # ----- evaluation -----

    def survival_curve(self, x):
        logits, _ = self.backbone.apply(self.state.params, jnp.asarray(x))
        return survival_curve(logits, axis=self.cfg.axis)

    def evaluate(self, x, ts, cs, train_ts=None, train_cs=None):
        """Return (ci, ci_ipcw, ibs, ibs_ipcw) on a single batch.

        IBS is computed on ``surv[:, 1:]`` so the horizon index `h=1..t_max`
        maps to ``surv[:, h-1] = P(T > h)`` rather than the constant
        ``P(T > 0) = 1`` leading column (a legacy off-by-one). Pass
        ``train_ts/train_cs`` to fit the IPCW censoring KM on the training
        set (recommended for held-out evaluation).
        """
        surv = np.asarray(self.survival_curve(x))
        scores = np.asarray(median_survival_time(surv))
        ts_np, cs_np = np.asarray(ts), np.asarray(cs)
        ci = float(concordance_index(scores, ts, cs))
        ci_ipcw = concordance_index_ipcw(scores, ts_np, cs_np, train_ts, train_cs)
        ibs = integrated_brier_score_np(surv[:, 1:], ts_np, cs_np)
        ibs_ipcw = integrated_brier_score_ipcw(surv[:, 1:], ts_np, cs_np,
                                               train_ts, train_cs)
        return ci, ci_ipcw, ibs, ibs_ipcw

    def evaluate_landmarks(self, x, ts, cs, landmarks, horizons=None,
                           train_ts=None, train_cs=None):
        """Dynamic-DeepHit style evaluation on a landmark x horizon grid.

        At landmark ``t_M`` the model's hazard row ``logits[:, t_M, :]`` is
        already the distribution conditional on surviving to ``t_M``, so the
        risk within ``delta`` steps is ``1 - prod_{k<=delta}(1 - h_k)`` --
        the same quantity the reference computes by renormalising its
        cumulative incidence over ``[t_M, t_M + delta]``.

        Only subjects still at risk at ``t_M`` are scored, with time measured
        forward from the landmark. Horizons default to the 25/50/75th
        percentiles of remaining time among at-risk TRAINING subjects, the
        generic analogue of the reference's domain-chosen 1/3/5/10-year
        windows. Returns ``{(t_M, delta): {...metrics...}}``.
        """
        logits, _ = self.backbone.apply(self.state.params, jnp.asarray(x))
        haz = np.asarray(jax.nn.sigmoid(logits))
        ts, cs = np.asarray(ts), np.asarray(cs).astype(bool)
        tr_ts = np.asarray(train_ts) if train_ts is not None else None
        tr_cs = np.asarray(train_cs).astype(bool) if train_cs is not None else None

        out = {}
        for t_m in landmarks:
            t_m = int(t_m)
            if t_m >= haz.shape[1]:
                continue
            at_risk = ts > t_m
            if at_risk.sum() < 10:
                continue
            rem = (ts - t_m)[at_risk]
            rem_cs = cs[at_risk]
            # Survival from the landmark, accumulated in log space and in
            # float64: a plain cumprod underflows to 0.0 at long horizons, so
            # every subject's risk ties at exactly 1.0 and the C(t)-index
            # collapses to 0 from ties rather than from bad ranking.
            h_lm = np.clip(haz[at_risk, t_m, :].astype(np.float64), 0.0, 1 - 1e-12)
            surv = np.exp(np.cumsum(np.log1p(-h_lm), axis=1))

            tr_rem = tr_cs_r = None
            if tr_ts is not None:
                tr_at_risk = tr_ts > t_m
                if tr_at_risk.sum() >= 10:
                    tr_rem = (tr_ts - t_m)[tr_at_risk]
                    tr_cs_r = tr_cs[tr_at_risk]

            if horizons is None:
                base = tr_rem if tr_rem is not None else rem
                hs = np.unique(np.percentile(base, [25, 50, 75]).astype(int))
                hs = [h for h in hs if 1 <= h <= surv.shape[1]]
            else:
                hs = [int(h) for h in horizons if 1 <= h <= surv.shape[1]]

            for d in hs:
                risk = 1.0 - surv[:, d - 1]
                out[(t_m, d)] = {
                    "n_at_risk": int(at_risk.sum()),
                    "n_events_by_h": int(((rem <= d) & ~rem_cs).sum()),
                    "td_ci": td_concordance_index(risk, rem, rem_cs, d),
                    "td_bs": td_brier_score(risk, rem, rem_cs, d),
                    "td_ci_ipcw": td_concordance_index(risk, rem, rem_cs, d,
                                                       tr_rem, tr_cs_r),
                    "td_bs_ipcw": td_brier_score(risk, rem, rem_cs, d,
                                                 tr_rem, tr_cs_r),
                }
        return out

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
