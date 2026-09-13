"""Survival metrics: concordance index, Kaplan-Meier, Brier score."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from lifelines.utils import concordance_index as _ci
from sksurv.metrics import concordance_index_ipcw as _ci_ipcw
from sksurv.metrics import integrated_brier_score as _ibs_ipcw
from sksurv.util import Surv


def concordance_index(scores, ts, cs):
    """Concordance index. `cs` is the censoring indicator (True = censored)."""
    cs = cs.astype(jnp.bool_)
    return _ci(ts + cs, scores, ~cs)


def _surv_struct(ts, cs):
    return Surv.from_arrays(event=~np.asarray(cs).astype(bool),
                            time=np.asarray(ts).astype(float))


def _prep(ts, cs, train_ts, train_cs, surv=None):
    """Restrict eval to [0, tau) where tau = max training time so sksurv's
    censoring-KM lookup stays in-support. Also returns the tau bound."""
    tr_ts = np.asarray(train_ts if train_ts is not None else ts).astype(float)
    tr_cs = np.asarray(train_cs if train_cs is not None else cs).astype(bool)
    tau = float(tr_ts.max())
    ts_np = np.asarray(ts).astype(float)
    keep = ts_np < tau
    test = _surv_struct(ts_np[keep], np.asarray(cs)[keep])
    train = _surv_struct(tr_ts, tr_cs)
    return train, test, keep, tau, (np.asarray(surv)[keep] if surv is not None else None)


def concordance_index_ipcw(scores, ts, cs, train_ts=None, train_cs=None):
    """Uno's IPCW C-index (sksurv). `scores` follow `concordance_index`
    convention (higher = longer survival); we negate for risk."""
    try:
        train, test, keep, tau, _ = _prep(ts, cs, train_ts, train_cs)
        est = -np.asarray(scores)[keep]
        return float(_ci_ipcw(train, test, est, tau=tau)[0])
    except ValueError:
        # Degenerate follow-up range (e.g. tiny training subsets in the
        # learning-curve protocol) — IPCW estimate undefined.
        return float("nan")


def integrated_brier_score_ipcw(surv, ts, cs, train_ts=None, train_cs=None):
    """IPCW IBS (sksurv). `surv` shape (N, T) with column h = P(T > h)."""
    try:
        train, test, _, tau, surv = _prep(ts, cs, train_ts, train_cs, surv)
        times = np.arange(1, surv.shape[1] + 1, dtype=float)
        lo = float(test["time"].min())
        hi = min(float(test["time"].max()), tau)
        keep_t = (times > lo) & (times < hi)
        return float(_ibs_ipcw(train, test, surv[:, keep_t], times[keep_t]))
    except ValueError:
        # Degenerate follow-up range (see concordance_index_ipcw).
        return float("nan")


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
