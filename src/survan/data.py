"""Dataset loaders, splits, and batch iteration."""

from __future__ import annotations

import os
from math import ceil
from pickle import load
from typing import Any, Mapping

import h5py
import jax.numpy as jnp
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ---------- target/mask construction ----------


def _pad_to(x, shape):
    a1, a2 = x.shape
    b1, b2 = shape
    assert b2 >= a2 and b1 >= a1
    res = np.hstack((x, np.zeros((a1, b2 - a2), dtype=x.dtype)))
    res = np.vstack((res, np.zeros((b1 - a1, b2), dtype=x.dtype)))
    return res


def _single_target_and_mask(seq, t, c, landmark=False):
    # float32/bool throughout: the values are exact 0/1, and float64
    # intermediates quadruple peak memory on big_rw (n=10k, h=100).
    h, _ = seq.shape
    target = np.zeros((h, h), dtype=np.float32)
    h_ws = np.ones((h, h), dtype=np.float32)
    mask = np.ones_like(target)
    if not c:  # event observed within horizon
        target = np.eye(t, dtype=np.float32)[::-1]
        target = _pad_to(target, shape=(h, h))
        if landmark:
            tt = t.item() if isinstance(t, np.ndarray) else t
            if tt <= h:
                h_ws = np.tril(np.ones_like(target), -(h - t))[::-1]
                mask[t:, :] = 0
        else:
            t_aux = min(t, seq.shape[0])
            h_ws = _pad_to(np.ones((1, t_aux), dtype=np.float32), shape=(h, h))
            mask[1:, :] = 0
    return target, h_ws, mask


def get_targets_and_masks(seqs, ts, cs, landmark):
    targets, h_ws, masks = [], [], []
    for seq, t, c in zip(seqs, ts, cs):
        y, w, m = _single_target_and_mask(seq, t, c, landmark=landmark)
        targets.append(y)
        h_ws.append(w)
        masks.append(m)
    return (
        np.stack(targets),
        np.stack(h_ws).astype(bool),
        np.stack(masks).astype(bool),
    )


# ---------- dataset loaders ----------


def _load_pickle(data_path, horizon=None):
    data = load(open(data_path, "rb"))
    return np.array(data["seqs"]), np.array(data["ts"]), np.array(data["cs"])


def _load_h5(data_path, horizon=None, normalize=False):
    with h5py.File(data_path, "r") as f:
        seqs = np.array(f["seqs"]).astype(np.float32)
        ts = np.array(f["ts"])
        cs = np.array(f["cs"])
    if normalize:
        means = np.nanmean(seqs, axis=(0, 1))
        means = np.where(np.isnan(means), 0.0, means)
        stds = np.nanstd(seqs, axis=(0, 1))
        stds = np.where((stds < 1e-12) | np.isnan(stds), 1.0, stds)
        seqs = np.where(np.isnan(seqs), means[None, None, :], seqs)
        seqs = (seqs - means) / stds
    return seqs, ts, cs


def _load_lastfm_months(data_path, horizon=None, normalize=True):
    logs = pd.read_csv(os.path.join(data_path, "surv_logs_last.csv"))
    logs = logs.drop(columns=["Unnamed: 0"])
    events = pd.read_csv(os.path.join(data_path, "events.csv"))
    ts = events.time.values
    cs = events.censored.values.astype(bool)
    horizon = np.max(ts) if horizon is None else horizon

    cols = [
        "listen_count", "unique_artists", "artist_entropy", "repeat_ratio",
        "novelty", "delta_listen_count", "delta_unique_artists",
        "delta_artist_entropy", "delta_repeat_ratio", "delta_novelty",
    ]
    if normalize:
        logs[cols] = StandardScaler().fit_transform(logs[cols])

    def pad(group):
        vals = group[cols].values
        pad_w = max(0, horizon - len(group))
        return np.pad(vals, ((0, pad_w), (0, 0)))

    seqs = np.stack(logs.groupby("userid").apply(pad).values, axis=0)
    return seqs, ts - cs.astype(int), cs


def load_preprocessed(data_path):
    """Load an HDF5 with precomputed targets/masks/weights."""
    with h5py.File(data_path, "r") as f:
        return (
            np.array(f["seqs"]),
            np.array(f["ts"]),
            np.array(f["cs"]).astype(bool),
            np.array(f["h_tgt"]),
            np.array(f["h_ws"]).astype(bool),
            np.array(f["mask"]).astype(bool),
        )


_LOADERS = {
    "aids": _load_pickle,
    "pbc2": _load_pickle,
    "big_rw": _load_pickle,
    "rw": _load_pickle,
    "scania": _load_pickle,
    "churn_lastfm_months": _load_lastfm_months,
    "nasa": lambda **kw: _load_h5(**kw, normalize=False),
    "mimic": lambda **kw: _load_h5(**kw, normalize=True),
}


def load_dataset(
    name: str,
    landmark: bool,
    compute_targets: bool,
    kwargs: Mapping[str, Any],
):
    """Load `(seqs, ts, cs, target, h_ws, mask)`.

    Targets/weights/mask are precomputed when `compute_targets=True`; else None.
    """
    if name not in _LOADERS:
        raise KeyError(f"Unknown dataset {name!r}")
    seqs, ts, cs = _LOADERS[name](**kwargs)
    seqs = seqs.astype(np.float32)
    target, h_ws, mask = (
        get_targets_and_masks(seqs, ts, cs, landmark)
        if compute_targets else (None, None, None)
    )
    return seqs, ts, cs, target, h_ws, mask


# ---------- splits ----------


def train_val_test_split(arrays, ts, cs, seed, test_size, val_size=None, stratify=False):
    """Stratified (by censoring) or random split. Returns dict keyed by split.

    Uses `np.random.seed(seed)` + `np.random.shuffle` to match the legacy
    ordering bit-for-bit so historical result paths can be reproduced.
    """
    n = len(ts)
    np.random.seed(seed)
    if stratify:
        idx_pos = np.where(cs.astype(bool) == False)[0]
        idx_neg = np.where(cs.astype(bool) == True)[0]
        np.random.shuffle(idx_pos)
        np.random.shuffle(idx_neg)

        def cut(idx, frac):
            return int(len(idx) * frac)

        n_te_p, n_te_n = cut(idx_pos, test_size), cut(idx_neg, test_size)
        n_va_p = cut(idx_pos, val_size) if val_size else 0
        n_va_n = cut(idx_neg, val_size) if val_size else 0
        te = np.concatenate([idx_pos[:n_te_p], idx_neg[:n_te_n]])
        va = np.concatenate([idx_pos[n_te_p:n_te_p + n_va_p],
                             idx_neg[n_te_n:n_te_n + n_va_n]]) if val_size else None
        tr = np.concatenate([idx_pos[n_te_p + n_va_p:], idx_neg[n_te_n + n_va_n:]])
    else:
        idx = np.arange(n)
        np.random.shuffle(idx)
        n_te = int(n * test_size)
        n_va = int(n * val_size) if val_size else 0
        te = idx[:n_te]
        va = idx[n_te:n_te + n_va] if val_size else None
        tr = idx[n_te + n_va:]

    splits = {"train": tr, "test": te}
    if va is not None:
        splits["val"] = va

    def take(a, ix):
        return None if a is None else a[ix]

    out = {}
    for split, ix in splits.items():
        out[split] = {k: take(v, ix) for k, v in arrays.items()}
        out[split]["ts"] = ts[ix]
        out[split]["cs"] = cs[ix]
    return out


# ---------- batch iterator ----------


class BatchIterator:
    """Yields dict batches. Re-shuffles on every `reset()`.

    Uses the global `np.random` state for shuffles to bit-match the legacy
    training trajectory (the legacy generators called `np.random.shuffle`
    on a per-epoch permutation without a private RNG).
    """

    def __init__(self, arrays: Mapping[str, np.ndarray], batch_size: int,
                 shuffle: bool = True):
        self.arrays = {k: v for k, v in arrays.items() if v is not None}
        ref = next(iter(self.arrays.values()))
        self.n = len(ref)
        self.X = self.arrays.get("X", ref)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._gen = self._iter()

    def __iter__(self):
        return self

    def __len__(self):
        return ceil(self.n / self.batch_size)

    def reset(self):
        self._gen = self._iter()

    def __next__(self):
        return next(self._gen)

    def _iter(self):
        idx = np.arange(self.n)
        if self.shuffle:
            np.random.shuffle(idx)
        for i in range(0, self.n, self.batch_size):
            sl = idx[i:i + self.batch_size]
            yield {k: v[sl] for k, v in self.arrays.items()}

    # legacy accessors used by eval()
    @property
    def ts(self):
        return self.arrays["ts"]

    @property
    def cs(self):
        return self.arrays["cs"]


def to_jax(*arrays):
    return tuple(jnp.asarray(a) for a in arrays)
