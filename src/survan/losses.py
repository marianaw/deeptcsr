"""Loss functions and TC target/weight construction."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax


# ---------- TC soft targets & weights (lifted from deep_lambda_cox) ----------


def tc_targets(b_tgt, h_tgt, lambda_, T):
    """Soft hazard targets blended with the target-network forecast.

    Args:
        b_tgt: target-network sigmoid forecasts, shape (B, T, T).
        h_tgt: hard one-hot targets, shape (B, T, T).
        lambda_: blending coefficient.
        T: horizon.
    """
    B = b_tgt.shape[0]
    stgt_init = jnp.roll(b_tgt[:, T - 1], -1).at[:, T - 1].set(0.0)
    htgt_init = jnp.zeros((B, T))

    def step(carry, h):
        next_b, next_h = carry
        ht, bt = h
        out = lambda_ * next_h + (1 - lambda_) * next_b
        out = jnp.roll(out, 1).at[:, 0].set(ht[:, 0])
        cond = ht[:, 0] == 1
        out = jnp.where(cond[:, None], jnp.zeros((B, T)).at[:, 0].set(1), out)
        return (bt, out), out

    _, out = jax.lax.scan(
        step, (stgt_init, htgt_init),
        (jnp.transpose(h_tgt, (1, 0, 2)), jnp.transpose(b_tgt, (1, 0, 2))),
        reverse=True,
    )
    return jnp.transpose(out, (1, 0, 2))


def tc_weights(s_wgt, h_wgt, h_tgt, c, lambda_, T):
    """Soft importance weights, blended with the target-network forecast."""
    B = s_wgt.shape[0]
    w_init = jnp.roll(s_wgt[:, T - 1], 1).at[:, 0].set(1.0)
    w_init = jnp.where(c[:, None], jnp.ones((B, T)), w_init)

    def step(carry, h):
        next_s, next_h = carry
        ht, hw, sw = h
        out = lambda_ * next_h + (1 - lambda_) * next_s
        val = jax.lax.select(c, jnp.ones_like(hw[:, 0], dtype=jnp.float32),
                             hw[:, 0].astype(jnp.float32))
        out = jnp.roll(out, 1).at[:, 0].set(val)
        cond = ht[:, 0] == 1
        out = jnp.where(cond[:, None], jnp.zeros((B, T)).at[:, 0].set(1), out)
        return (sw, out), out

    _, out = jax.lax.scan(
        step, (s_wgt[:, T - 1], w_init),
        (jnp.transpose(h_tgt, (1, 0, 2)),
         jnp.transpose(h_wgt, (1, 0, 2)),
         jnp.transpose(s_wgt, (1, 0, 2))),
        reverse=True,
    )
    return jnp.transpose(out, (1, 0, 2))


def target_survival_weights(tgt_logits):
    """Target-network survival product, shifted by 1, used as TC importance weights."""
    log_h = jax.nn.log_sigmoid(tgt_logits)
    s = jnp.exp(jnp.cumsum(log_h - tgt_logits, axis=-1))
    return jnp.roll(s, 1).at[:, :, 0].set(1.0)


# ---------- hazard / ranking / prediction losses ----------


def hazard_bce(logits, targets, mask, weights=None, mode="mean"):
    """Masked (and optionally TC-weighted) BCE over the hazard logits.

    mode="mean": average over all elements (Cox / TC-Cox legacy).
    mode="weighted": divide by the sum of weights (DDH legacy).
    """
    bce = optax.sigmoid_binary_cross_entropy(logits, targets)
    w = mask.astype(jnp.float32)
    if weights is not None:
        w = w * weights
    if mode == "mean":
        return jnp.mean(bce * w)
    if mode == "weighted":
        return jnp.sum(bce * w) / jnp.maximum(jnp.sum(w), 1.0)
    raise ValueError(f"mode must be 'mean' or 'weighted', got {mode!r}")


def _hazard_path(logits, axis):
    h = jax.nn.sigmoid(logits)
    if axis == 2:
        return h[:, 0, :]
    if axis == 1:
        return h
    raise ValueError(f"axis must be 1 or 2, got {axis}")


def _surv_and_cdf(logits, axis):
    """Per-sample (survival, cdf) along the horizon."""
    hp = _hazard_path(logits, axis)  # (B, H)

    def step(prev, ht):
        return prev * (1 - ht), (prev * (1 - ht), prev * ht)

    init = jnp.ones(hp.shape[0], dtype=hp.dtype)
    _, (s, e) = jax.lax.scan(step, init, hp.T)
    surv = jnp.concatenate([jnp.ones((1, hp.shape[0])), s], axis=0).T  # (B, H+1)
    cdf = jnp.cumsum(e.T, axis=1)
    return surv, cdf


def survival_curve(logits, axis):
    surv, _ = _surv_and_cdf(logits, axis)
    return surv


def ranking_loss(logits, ts, cs, axis, sigma):
    _, cdf = _surv_and_cdf(logits, axis)
    if cdf.shape[1] == 0:
        return jnp.array(0.0, dtype=jnp.float32)

    ts = jnp.asarray(ts, dtype=jnp.int32)
    events = (~cs.astype(jnp.bool_)).astype(jnp.float32)
    idx = jnp.clip(ts, 1, cdf.shape[1]) - 1

    cdf_at = jax.vmap(lambda k: cdf[:, k])(idx)  # (B, B)
    cdf_self = jnp.take_along_axis(cdf, idx[:, None], axis=1).squeeze(1)
    diff = cdf_self[:, None] - cdf_at
    rank = jnp.exp(-diff / sigma)

    time_mask = ts[None, :] > ts[:, None]
    valid = ((events[:, None] == 1.0) & time_mask).astype(jnp.float32)
    return jnp.sum(rank * valid) / jnp.maximum(jnp.sum(valid), 1.0)


def covariate_prediction_loss(preds, inputs):
    """Next-step covariate reconstruction loss (DDH auxiliary head)."""
    p = preds[:, :-1]
    t = inputs[:, 1:]
    if p.shape[1] == 0:
        return jnp.array(0.0, dtype=preds.dtype)
    sq = jnp.sum((p - t) ** 2, axis=-1)
    return jnp.sum(sq) / (p.shape[0] * p.shape[1] * p.shape[2])


def median_survival_time(surv):
    """First horizon index where survival drops to 0.5 (or last index if never)."""
    def _idx(curve):
        crosses = curve <= 0.5
        return jnp.where(jnp.any(crosses), jnp.argmax(crosses),
                         curve.shape[0] - 1).astype(jnp.float32)
    return jax.vmap(_idx)(surv)
