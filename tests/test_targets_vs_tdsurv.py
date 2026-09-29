"""Our TC targets/weights must match the reference tdsurv implementation
(Maystre & Russo) on data with mid-sequence AND horizon censoring.

    TDSURV_LIB=/path/to/tdsurv/lib uv run python tests/test_targets_vs_tdsurv.py
"""
import os
import sys

import jax
import numpy as np

sys.path.insert(0, os.environ.get("TDSURV_LIB", os.path.join(
    os.path.dirname(__file__), "..", "..", "tdsurv", "lib")))
from tdsurv import CoxPH  # noqa: E402
from tdsurv.utils import unroll  # noqa: E402

from survan.data import get_targets_and_masks
from survan.losses import tc_targets, tc_weights, target_survival_weights

H, D, N = 6, 3, 60


def toy(rng):
    """Project convention: ts = n_steps - censored."""
    cs = rng.uniform(size=N) < 0.5
    n_steps = np.where(cs, rng.integers(2, H + 1, N), rng.integers(1, H + 1, N))
    seqs = np.zeros((N, H, D), dtype=np.float32)
    for i, n in enumerate(n_steps):
        seqs[i, :n] = rng.normal(size=(n, D))
    return seqs, n_steps - cs.astype(int), cs


def ours(seqs, ts, cs, params, lam):
    y, h_ws, mask = get_targets_and_masks(seqs, ts, cs, landmark=True)
    logits = (seqs @ params[:D])[..., None] + params[D:]  # == tdsurv CoxPH
    w = tc_weights(target_survival_weights(logits), h_ws, y, cs, lam, H, ts)
    return np.asarray(tc_targets(jax.nn.sigmoid(logits), y, lam, H, cs, ts)), \
        np.asarray(w) * mask, mask


def reference(seqs, ts, cs, params, lam):
    model = CoxPH(H, D)
    model.params = params
    mult = lam ** np.arange(H)
    mult[:-1] *= 1 - lam
    useqs, uts, ucs = unroll(seqs, ts, cs)
    ys, ws = np.zeros((len(uts), H)), np.zeros((len(uts), H))
    for m, mu in enumerate(mult, start=1):
        if mu:
            y_m, w_m = model._targets(m, useqs, uts, ucs)
            ys += mu * np.asarray(y_m)
            ws += mu * np.asarray(w_m)
    # unroll order: row 0 of every seq, then row i of seqs with ts > i, ...
    rows = [(n, 0) for n in range(len(ts))]
    for i in range(1, ts.max()):
        rows += [(n, i) for n in np.flatnonzero(ts > i)]
    return ys, ws, rows


def check(lam, seed=0):
    rng = np.random.default_rng(seed)
    seqs, ts, cs = toy(rng)
    params = rng.normal(size=D + H).astype(np.float32)
    y, w, mask = ours(seqs, ts, cs, params, lam)
    assert (mask[:, :, 0].astype(bool) == (np.arange(H) < ts[:, None])).all(), \
        "trained rows must be exactly the states with an observed successor"
    ry, rw, rows = reference(seqs, ts, cs, params, lam)
    bad = []
    for r, (n, i) in enumerate(rows):
        ok = np.allclose(w[n, i], rw[r], atol=1e-5) and np.allclose(
            (y[n, i] * w[n, i]), ry[r] * rw[r], atol=1e-5)
        if not ok:
            bad.append((n, i, bool(cs[n])))
    return bad


def test_lambda_0():
    assert not check(0.0)


def test_lambda_1():
    assert not check(1.0)


def test_lambda_half():
    assert not check(0.5)


if __name__ == "__main__":
    fails = 0
    for lam in (0.0, 0.5, 0.95, 1.0):
        bad = check(lam)
        n_cens = sum(c for *_, c in bad)
        print(f"lambda={lam}: {len(bad)} mismatched rows "
              f"({n_cens} censored, {len(bad) - n_cens} events)")
        fails += len(bad)
    sys.exit(fails > 0)
