from pickle import load
import jax.numpy as jnp

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
        mask = jnp.tril(jnp.ones_like(seq), -(t-1)).astype(jnp.bool_)[::-1]
    else:
        mask = jnp.ones((1, t))
        mask = pad_to(mask, seq).astype(jnp.bool_)
    return target, mask


def get_data(data_path, landmark):
    data = load(open(data_path, 'rb'))
    seqs = data['seqs']
    cs = data['cs']
    ts = data['ts']

    masks = []
    targets = []
    for seq, t, c in zip(seqs, ts, cs):
        target, mask = get_target_and_mask(seq, t, c, landmark=landmark)
        targets.append(target)
        masks.append(mask)
    
    target = jnp.stack(targets)
    mask = jnp.stack(masks)

    return seqs, target, mask
