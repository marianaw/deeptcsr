from math import ceil
import os
import h5py
from pickle import load
import jax
import jax.numpy as jnp
from lifelines.utils import concordance_index as _concordance_index


def pad_to(x1, x2):
    a1, a2 = x1.shape
    b1, b2 = x2.shape
    assert b2 >= a2 and b1 >= a1

    miss_cols = b2 - a2
    miss_rows = b1 - a1

    res = jnp.hstack((x1, jnp.zeros((a1, miss_cols))))
    res = jnp.vstack((res, jnp.zeros((miss_rows, b2))))
    return res

def get_target_and_mask(seq, t, c, landmark=False):
    target = jnp.zeros_like(seq)
    if not c:
        target = jnp.eye(t)[::-1]
        target = pad_to(target, seq)
    if landmark:
        # import ipdb;ipdb.set_trace()
        mask = jnp.tril(jnp.ones_like(seq), -(t.item()-1))[::-1]
    else:
        mask = jnp.ones((1, t))
        mask = pad_to(mask, seq)
    return target, mask


def pad_sequences(seqs, max_length):
    num_sequences = len(seqs)
    dim = seqs[0].shape[-1]
    padded_sequences = jnp.full((num_sequences, max_length, dim), 0.0, dtype=jnp.float32)

    for i, sequence in enumerate(seqs):
        length = jnp.minimum(sequence.shape[0], max_length)
        padded_sequences = padded_sequences.at[i, :length, :].set(sequence[:length])

    return padded_sequences


def get_data(dataset_name, kwargs):
    loaders = {'aids': get_data_baseline,
               'single_task': get_single_task_dataset,
               }
    
    try:
        return loaders[dataset_name](**kwargs)
    except KeyError:
        raise Exception('type of dataset not found.')


def get_single_task_dataset(task_id, data_path, horizon, pad=True):
    seqs = []
    for root, dirs, files in os.walk(data_path):
        for filename in files:
            if filename.endswith('.mat') and 'task_{}'.format(task_id) in filename:
                file_path = os.path.join(root, filename)
                with h5py.File(file_path, 'r') as f:
                    if 'traces_self' in f:
                        size = f['traces_self'].shape[0]
                        # Extract the array under the 'traces_self' key
                        t_self = [f['traces_self'][i].reshape(-1, 39) for i in range(size)]
                        seqs.extend(t_self)

    ts = jnp.array([len(arr) for arr in seqs])
    cs = jnp.where(ts > horizon, 1, 0)
    ts = ts - cs.astype(jnp.int32)
    if pad:
        seqs = pad_sequences(seqs, horizon)
    
    return seqs, ts, cs


def get_data_baseline(data_path, landmark):
    data = load(open(data_path, 'rb'))
    seqs = jnp.array(data['seqs'])
    cs = jnp.array(data['cs'])
    ts = jnp.array(data['ts'])

    masks = []
    targets = []
    for seq, t, c in zip(seqs, ts, cs):
        target, mask = get_target_and_mask(seq, t, c, landmark=landmark)
        targets.append(target)
        masks.append(mask)
    
    target = jnp.stack(targets)
    mask = jnp.stack(masks)

    return seqs, target, mask, ts, cs


def train_test_split(X, ts, cs, rng, test_size=0.2):
    # Shuffle the indices of the data
    num_samples = X.shape[0]
    shuffled_indices = jax.random.permutation(rng, jnp.arange(num_samples))

    # Calculate the number of samples in the test set
    num_test_samples = int(num_samples * test_size)

    # Split the shuffled indices into train and test sets
    test_indices = shuffled_indices[:num_test_samples]
    train_indices = shuffled_indices[num_test_samples:]

    # Use the indices to split the data
    X_train = X[train_indices]
    X_test = X[test_indices]
    ts_train = ts[train_indices]
    ts_test = ts[test_indices]
    cs_train = cs[train_indices]
    cs_test = cs[test_indices]

    return X_train, X_test, \
        ts_train, ts_test, cs_train, cs_test


class DataGenerator:

    def __init__(self, X, y, mask, ts, cs, batch_size, rng, shuffle=True):
        self.X = X
        self.y = y
        self.ts = ts
        self.cs = cs
        self.mask = mask
        self.shuffle = shuffle
        self.rng = rng
        self.batch_size = batch_size
        self.generator = self.batch_generator()

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
            permutation = jax.random.permutation(subkey, jnp.arange(num_samples))
            X = X[permutation]
            y = y[permutation]
            mask = mask[permutation]

        for i in range(0, num_samples, batch_size):
            batch_X = X[i:i + batch_size]
            batch_y = y[i:i + batch_size]
            batch_m = mask[i:i + batch_size]
            yield batch_X, batch_y, batch_m
    
    def __iter__(self):
        return self
    
    def __len__(self):
        return ceil(len(self.X)/self.batch_size)
    
    def reset(self):
        self.generator = self.batch_generator()
    
    def __next__(self):
        # try:
        batch_X, batch_y, batch_m = next(self.generator)
        return batch_X, batch_y, batch_m
        # except StopIteration:
            
            # batch_X, batch_y, batch_m = next(self.generator)
            # return batch_X, batch_y, batch_m


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
        return (seqs_[:,:2], ts_, cs_)
    return (seqs_, ts_, cs_)
