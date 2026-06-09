"""Survival metrics: concordance index, Kaplan-Meier, Brier score."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from lifelines.utils import concordance_index as _ci


def concordance_index(scores, ts, cs):
    """Concordance index. `cs` is the censoring indicator (True = censored)."""
    cs = cs.astype(jnp.bool_)
    return _ci(ts + cs, scores, ~cs)


def kaplan_meier(ts, cs):
    """KM survival estimator. `cs` is True for censored observations."""
    cs = cs.astype(jnp.bool_)
    steps = jnp.arange(0, jnp.max(ts) + 1)
    at_risk = jnp.sum(ts[:, None] >= steps, axis=0)
    events = jnp.sum(ts[~cs, None] == steps, axis=0)
    return jnp.cumprod(1 - events / at_risk)


def brier_score(h, surv, ts, cs):
    """Time-dependent Brier score at horizon `h` (1-indexed)."""
    cs = cs.astype(jnp.bool_)
    ws = kaplan_meier(ts - ~cs, ~cs)

    # observed events at or before h
    mask_e = jnp.where((ts <= h) & ~cs, 1, 0)
    a = (1 / ws[ts - 1]) * surv[:, h - 1] ** 2 * mask_e
    a = jnp.where(jnp.isfinite(a), a, 0)

    # still active beyond h
    mask_c = jnp.where((ts > h) | ((ts == h) & cs), 1, 0)
    b = (1 / ws[h - 1]) * (1.0 - surv[:, h - 1]) ** 2 * mask_c
    b = jnp.where(jnp.isfinite(b), b, 0)

    return jnp.sum(a) + jnp.sum(b)


def integrated_brier_score(surv, ts, cs):
    f = jax.vmap(partial(brier_score, surv=surv, ts=ts, cs=cs))
    t_max = jnp.max(ts)
    hs = jnp.arange(1, t_max + 1)
    return jnp.sum(f(hs) / (t_max * len(ts)))


# numpy variants for use during evaluation where JIT is unhelpful
def _km_np(ts, cs):
    cs = cs.astype(np.bool_)
    steps = np.arange(0, np.max(ts) + 1)
    at_risk = np.sum(ts[:, None] >= steps, axis=0)
    events = np.sum(ts[~cs, None] == steps, axis=0)
    return np.cumprod(1 - events / at_risk)


def integrated_brier_score_np(surv, ts, cs):
    """NumPy IBS for survival curves with shape (N, T)."""
    cs = cs.astype(np.bool_)
    t_max = int(np.max(ts))
    if t_max == 0:
        return 0.0
    ws = _km_np(ts - ~cs, ~cs)
    T = surv.shape[1]
    total = 0.0
    for h in range(1, t_max + 1):
        if h > T:
            break
        mask_e = ((ts <= h) & ~cs).astype(float)
        a = (1 / ws[np.clip(ts, 1, T) - 1]) * surv[:, h - 1] ** 2 * mask_e
        a = np.where(np.isfinite(a), a, 0)
        mask_c = ((ts > h) | ((ts == h) & cs)).astype(float)
        b = (1 / ws[h - 1]) * (1.0 - surv[:, h - 1]) ** 2 * mask_c
        b = np.where(np.isfinite(b), b, 0)
        total += np.sum(a) + np.sum(b)
    return float(total / (t_max * len(ts)))
