from functools import partial
from math import ceil
import os
import h5py
from pickle import load
import numpy as np
import jax
import jax.numpy as jnp
from lifelines.utils import concordance_index as _concordance_index


def pad_to(x1, shape):

    a1, a2 = x1.shape
    b1, b2 = shape

    assert b2 >= a2 and b1 >= a1

    miss_cols = b2 - a2
    miss_rows = b1 - a1

    res = jnp.hstack((x1, jnp.zeros((a1, miss_cols))))
    res = jnp.vstack((res, jnp.zeros((miss_rows, b2))))
    return res


def get_single_target_and_mask(seq, t, c, landmark=False):
    h, _ = seq.shape
    target = jnp.zeros((h, h))
    h_ws = jnp.ones((h, h))
    mask = jnp.ones_like(target)
    if not c:  # Subject reached terminal state within the horizon.
        target = jnp.eye(t)[::-1]
        target = pad_to(target, shape=(h, h))
        if landmark:
            h_ws = jnp.ones_like(target)
            tt = t.item()
            if tt <= h:
                h_ws = jnp.tril(jnp.ones_like(target), -(h-t.item()))[::-1]
                mask_out = h - t
                mask = mask.at[t:, :].set(jnp.zeros((mask_out, h)))
        else:
            t_aux = min(t, seq.shape[0])
            h_ws = jnp.ones((1, t_aux))
            h_ws = pad_to(h_ws, shape=(h, h))
            mask = mask.at[1:, :].set(jnp.zeros((h-1, h)))

    return target, h_ws, mask


def pad_sequences(seqs, max_length):
    num_sequences = len(seqs)
    dim = seqs[0].shape[-1]
    padded_sequences = jnp.full(
        (num_sequences, max_length, dim), 0.0, dtype=jnp.float32)

    for i, sequence in enumerate(seqs):
        length = jnp.minimum(sequence.shape[0], max_length)
        padded_sequences = padded_sequences.at[i, :length, :].set(
            sequence[:length])

    return padded_sequences


def get_data(dataset_name, landmark, kwargs):
    loaders = {'aids': get_data_baseline,
               'single_task': get_single_task_dataset,
               'mixed_tasks': get_mixed_task_dataset}

    try:
        seqs, ts, cs = loaders[dataset_name](**kwargs)
        target, h_ws, mask = get_targets_and_masks(seqs, ts, cs, landmark)
    except KeyError:
        raise Exception('type of dataset not found.')

    return seqs, ts, cs, target, h_ws, mask


def load_preprocessed_dataset(data_path):
    data = h5py.File(data_path, 'r')
    seqs = np.array(data['seqs'])
    ts = np.array(data['ts'])
    cs = np.array(data['cs'])
    h_ws = np.array(data['h_ws'])
    mask = np.array(data['mask'])
    h_tgt = np.array(data['h_tgt'])
    return seqs, ts, cs, h_tgt, h_ws, mask


def split_and_pad_last(arr, H=1000):
    t, dim = arr.shape
    n_splits = ceil(t/H)
    indices = np.arange(1, n_splits) * H
    arrs = np.array_split(arr, indices_or_sections=indices)
    last = arrs[-1]
    h, _ = last.shape
    if h < H:
        zs = np.zeros((H-h, dim))
        last = np.concatenate((last, zs))
    arr = np.stack(arrs[:-1] + [last])

    ts = t - indices
    ts = np.hstack((np.array([t]), ts))
    cs = np.hstack((np.ones_like(indices), np.array([0]))).astype(bool)
    return arr, ts, cs


def get_single_task_dataset(task_id, data_path, horizon=None, split=True, pad=False):
    seqs = []
    for root, dirs, files in os.walk(data_path):
        for filename in files:
            if filename.endswith('.mat') and 'task_{}'.format(task_id) in filename:
                file_path = os.path.join(root, filename)
                with h5py.File(file_path, 'r') as f:
                    if 'traces_self' in f:
                        size = f['traces_self'].shape[0]
                        # Extract the array under the 'traces_self' key
                        t_self = [f['traces_self']
                                  [i].reshape(-1, 39) for i in range(size)]
                        seqs.extend(t_self)

    if split:
        arrs, tss, css = [], [], []
        for seq in seqs:
            arr, ts, cs = split_and_pad_last(seq, horizon)
            arrs.append(arr)
            css.append(cs)
            tss.append(ts)
        seqs = np.vstack(arrs)
        ts = np.hstack(tss)
        cs = np.hstack(css)

    else:
        ts = np.array([len(arr) for arr in seqs])
        horizon = np.max(ts) if horizon is None else horizon
        cs = np.where(ts > horizon, 1, 0)
        pad = True

    if pad:
        seqs = pad_sequences(seqs, horizon)

    ts = ts - cs.astype(int)
    return seqs, ts, cs


def get_mixed_task_dataset(data_path, horizon=None, split=True, pad=False):
    seqs = []
    for root, dirs, files in os.walk(data_path):
        for filename in files:
            if filename.endswith('.mat'):
                file_path = os.path.join(root, filename)
                with h5py.File(file_path, 'r') as f:
                    if 'traces_self' in f:
                        size = f['traces_self'].shape[0]
                        # Extract the array under the 'traces_self' key
                        t_self = [f['traces_self']
                                  [i].reshape(-1, 39) for i in range(size)]
                        seqs.extend(t_self)

    if split:
        arrs, tss, css = [], [], []
        for seq in seqs:
            arr, ts, cs = split_and_pad_last(seq, horizon)
            arrs.append(arr)
            css.append(cs)
            tss.append(ts)
        seqs = np.vstack(arrs)
        ts = np.hstack(tss)
        cs = np.hstack(css)

    else:
        ts = np.array([len(arr) for arr in seqs])
        horizon = np.max(ts) if horizon is None else horizon
        cs = np.where(ts > horizon, 1, 0)
        pad = True

    if pad:
        seqs = pad_sequences(seqs, horizon)

    ts = ts - cs.astype(int)
    return seqs, ts, cs


def get_targets_and_masks(seqs, ts, cs, landmark):
    masks = []
    h_ws = []
    targets = []
    for seq, t, c in zip(seqs, ts, cs):
        target, h_w, mask = get_single_target_and_mask(
            seq, t, c, landmark=landmark)
        targets.append(target)
        h_ws.append(h_w)
        masks.append(mask)

    target = np.stack(targets)
    mask = np.stack(masks)
    h_ws = np.stack(h_ws)
    return target, h_ws, mask


def get_data_baseline(data_path, horizon=None):
    data = load(open(data_path, 'rb'))
    seqs = np.array(data['seqs'])
    cs = np.array(data['cs'])
    ts = np.array(data['ts'])

    return seqs, ts, cs


def train_test_split(X, target, h_ws, mask, ts, cs, test_size=0.2):
    # Shuffle the indices of the data
    num_samples = X.shape[0]
    shuffled_indices = np.arange(num_samples)
    np.random.shuffle(shuffled_indices)

    # Calculate the number of samples in the test set
    num_test_samples = int(num_samples * test_size)

    # Split the shuffled indices into train and test sets
    test_indices = shuffled_indices[:num_test_samples]
    train_indices = shuffled_indices[num_test_samples:]

    # Use the indices to split the data
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = target[train_indices]
    y_test = target[test_indices]
    hws_train = h_ws[train_indices]
    hws_test = h_ws[test_indices]
    m_train = mask[train_indices]
    m_test = mask[test_indices]
    ts_train = ts[train_indices]
    ts_test = ts[test_indices]
    cs_train = cs[train_indices]
    cs_test = cs[test_indices]

    return X_train, X_test, y_train, y_test, hws_train, hws_test, \
        m_train, m_test, ts_train, ts_test, cs_train, cs_test


class BaseDataGenerator:

    def __init__(self, X, ts, cs, y, mask, batch_size, rng, h_ws=None, shuffle=True):
        self.X = X
        self.y = y
        self.ts = ts
        self.cs = cs
        self.mask = mask
        self.shuffle = shuffle
        self.rng = rng
        self.batch_size = batch_size
        self.h_ws = h_ws
        self.generator = self.batch_generator()

    def batch_generator(self):
        raise NotImplementedError

    def __iter__(self):
        return self

    def __len__(self):
        return ceil(len(self.X)/self.batch_size)

    def reset(self):
        self.generator = self.batch_generator()

    def __next__(self):
        # try:
        batch = next(self.generator)
        return batch


class TgtMskDataGenerator(BaseDataGenerator):
    def batch_generator(self):
        X = self.X
        y = self.y
        mask = self.mask
        batch_size = self.batch_size
        rng = self.rng
        num_samples = X.shape[0]

        # Shuffle the data using the same random key for X and y if shuffle is True
        if self.shuffle:
            rng, subkey = jax.random.split(rng)
            permutation = jax.random.permutation(
                subkey, jnp.arange(num_samples))
            X = X[permutation]
            y = y[permutation]
            mask = mask[permutation]

        for i in range(0, num_samples, batch_size):
            batch_X = X[i:i + batch_size]
            batch_y = y[i:i + batch_size]
            batch_m = mask[i:i + batch_size]
            yield batch_X, batch_y, batch_m


class TimesDataGenerator(BaseDataGenerator):
    def batch_generator(self):
        X = self.X
        ys = self.y
        ts = self.ts
        cs = self.cs
        mask = self.mask
        h_ws = self.h_ws
        batch_size = self.batch_size
        rng = self.rng
        num_samples = X.shape[0]

        # Shuffle the data using the same random key for X and y if shuffle is True
        if self.shuffle:
            rng, subkey = jax.random.split(rng)
            permutation = jax.random.permutation(
                subkey, jnp.arange(num_samples))
            X = X[permutation]
            cs = cs[permutation]
            ts = ts[permutation]
            ys = ys[permutation]
            mask = mask[permutation]
            h_ws = h_ws[permutation]

        for i in range(0, num_samples, batch_size):
            batch_X = X[i:i + batch_size]
            batch_ts = ts[i:i + batch_size]
            batch_cs = cs[i:i + batch_size]
            batch_ys = ys[i:i + batch_size]
            batch_m = mask[i:i + batch_size]
            batch_hws = h_ws[i:i + batch_size]
            yield batch_X, batch_ts, batch_cs, batch_ys, batch_m, batch_hws


def batch_generator(X, y, mask, batch_size, rng, shuffle=True):
    num_samples = X.shape[0]

    # Shuffle the data using the same random key for X and y if shuffle is True
    if shuffle:
        rng, subkey = jax.random.split(rng)
        permutation = jax.random.permutation(subkey, jnp.arange(num_samples))
        X = X[permutation]
        y = y[permutation]
        mask = mask[permutation]

    for i in range(0, num_samples, batch_size):
        batch_X = X[i:i + batch_size]
        batch_y = y[i:i + batch_size]
        batch_m = mask[i:i + batch_size]
        yield batch_X, batch_y, batch_m


def kaplan_meier(ts, cs):
    """Kaplan-Meier estimator of survival curve."""
    cs = cs.astype(jnp.bool_)
    steps = jnp.arange(0, jnp.max(ts) + 1)
    # Number of individuals known to have survived up to step k = 0, 1, ...
    ns = jnp.sum(ts[:, jnp.newaxis] >= steps, axis=0)
    # Number of events that happened at step k = 0, 1, ...
    ds = jnp.sum(ts[~cs, jnp.newaxis] == steps, axis=0)
    # Product over k of (1 - empirical hazard at k).
    return jnp.cumprod(1 - ds / ns)


def concordance_index(scores, ts, cs):
    """Compute concordance-index for given scores."""
    # Thin wrapper around the `lifelines` implementation.
    cs = cs.astype(jnp.bool_)
    return _concordance_index(ts + cs, scores, ~cs)


def unroll(seqs, ts, cs, compress=False):
    """Unroll sequences.

    This function transforms each sequence `(x1, x2, ..., xt)` into
    subsequences `((x1, x2, ..., xt), (x2, ..., xt), ..., (xt,))`.

    Note: the smallest subsequence always contains two observed states, whether
    implicitly or explicitly.

    - For uncensored sequences, the smallest subsequence is `(xt,)` and
      implicitly accounts for the terminal state that follows.
    - For censored sequences, the smallest subsequence is `(x{t-1}, xt)`.
    """
    cs = cs.astype(jnp.bool_)
    seqs_ = jnp.copy(seqs)
    ts_ = jnp.copy(ts)
    cs_ = jnp.copy(cs)
    for i in range(1, jnp.max(ts)):
        idx = ts > i  # Indices of seqs whose successor state is observed.
        new = jnp.zeros((jnp.sum(idx),) + seqs.shape[1:], dtype=seqs.dtype)
        new = new.at[:, :-i].set(seqs[idx, i:])
        seqs_ = jnp.concatenate((seqs_, new))
        ts_ = jnp.concatenate((ts_, ts[idx] - i))
        cs_ = jnp.concatenate((cs_, cs[idx]))
    if compress:
        return (seqs_[:, :2], ts_, cs_)
    return (seqs_, ts_, cs_)


def score(beta, xs):
    return -jnp.dot(xs, beta)


def get_unroll_t(t, h):
    def cumsub(res, el):
        res = res - 1
        res = jax.nn.relu(res)
        return res, res

    taux = jnp.zeros(h).at[0].set(t)
    _, taux = jax.lax.scan(cumsub, t, taux)
    taux = jnp.insert(taux[:-1], 0, t)
    taux = taux.reshape(-1, 1)
    return taux


def get_unroll_ts(ts, h):
    partial_get_unroll_t = partial(get_unroll_t, h=h)
    get_ts = jax.vmap(partial_get_unroll_t)
    unrolled = get_ts(ts)
    return unrolled


def unroll_time(xs, ts, cs, ms, T):
    xs_ = xs.reshape(-1, xs.shape[-1])
    ts_ = get_unroll_ts(ts, T)
    ts_ = ts_.reshape(-1)
    cs_ = cs_ = cs.reshape(-1, 1, 1)
    cs_ = cs_.repeat(T, 1).reshape(-1)
    ms_ = ms.any(-1, keepdims=True)
    ms_ = ms_.reshape(-1)
    return xs_, ts_, cs_, ms_


def convert_to_jax_arrays(*numpy_arrays):
    jax_arrays = (jnp.asarray(arr) for arr in numpy_arrays)
    return jax_arrays
