"""Backbones for DeepTCSR.

Each backbone is a Haiku-callable with signature

    forward(x: (B, T, D)) -> (logits: (B, T, H), aux: Optional[Tensor])

`aux` carries auxiliary outputs (e.g. covariate-prediction head for DDH)
and is `None` for backbones that don't produce one.
"""

from __future__ import annotations

import haiku as hk
import jax
import jax.numpy as jnp


# ---------- TCN ----------


class _Chomp1D(hk.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def __call__(self, x):
        return x[:, :, :-self.chomp_size]


class _TemporalBlock(hk.Module):
    def __init__(self, n_out, kernel_size, dilation, padding, seed, dropout=0.2):
        super().__init__()
        self.conv1 = hk.Conv1D(n_out, kernel_size, padding=padding,
                               rate=dilation, data_format="NC...")
        self.chomp1 = _Chomp1D(padding[0])
        self.conv2 = hk.Conv1D(n_out, kernel_size, padding=padding,
                               rate=dilation, data_format="NC...")
        self.chomp2 = _Chomp1D(padding[0])
        self.dropout = dropout
        self.keys = hk.PRNGSequence(seed)

    def __call__(self, x):
        out = jax.nn.relu(self.chomp1(self.conv1(x)))
        out = hk.dropout(next(self.keys), self.dropout, out)
        out = jax.nn.relu(self.chomp2(self.conv2(out)))
        out = hk.dropout(next(self.keys), self.dropout, out)
        res = x if x.shape[1] == out.shape[1] else hk.Conv1D(
            out.shape[1], 1, data_format="NC...")(x)
        return jax.nn.relu(out + res)


def _tcn_forward(num_channels, kernel_size, dilation_factor, horizon, seed, dropout):
    def fwd(x):
        # x: (B, T, D) -> (B, D, T) for NC... conv format
        h = jnp.transpose(x, (0, 2, 1))
        for i, ch in enumerate(num_channels):
            dil = dilation_factor ** i
            pad = (kernel_size - 1) * dil
            h = _TemporalBlock(ch, kernel_size, dil, (pad, pad),
                               seed=seed + i, dropout=dropout)(h)
        h = jnp.transpose(h, (0, 2, 1))  # (B, T, C)
        return hk.Linear(horizon)(h), None
    return fwd


# ---------- Transformer ----------


def _positional_encoding(seq_len, hidden_size):
    pos = jnp.arange(seq_len)[:, None]
    div = jnp.exp(jnp.arange(0, hidden_size, 2) * -(jnp.log(10000.0) / hidden_size))
    pe = jnp.zeros((seq_len, hidden_size))
    pe = pe.at[:, 0::2].set(jnp.sin(pos * div))
    pe = pe.at[:, 1::2].set(jnp.cos(pos * div))
    return pe


def _transformer_forward(hidden_size, seq_len, horizon, num_layers, dropout,
                         seed, residual=True):
    """Causal transformer encoder over the state sequence.

    ``residual=True`` uses standard pre-norm blocks (x + attn(norm(x)),
    x + ffn(norm(x))). The original variant (``residual=False``) overwrote
    the hidden state at every layer with no skip path, so stacking layers
    degraded rather than helped -- in the architecture pilot it left the Cox
    family at chance level on Scania for every width/depth tried.
    """
    def fwd(x):
        keys = hk.PRNGSequence(seed)
        pe = _positional_encoding(seq_len, hidden_size)
        h = hk.Linear(hidden_size)(x) + pe
        # (1, 1, T, T) so it broadcasts over batch and heads.
        mask = jnp.tril(jnp.ones((seq_len, seq_len),
                                 dtype=jnp.float32))[None, None]
        w_init = hk.initializers.VarianceScaling(2 / num_layers)
        for _ in range(num_layers):
            if residual:
                a = hk.LayerNorm(axis=-1, create_scale=True,
                                 create_offset=True)(h)
                a = hk.MultiHeadAttention(num_heads=4, key_size=16,
                                          model_size=hidden_size,
                                          w_init=w_init)(a, a, a, mask=mask)
                h = h + hk.dropout(next(keys), dropout, a)
                f = hk.LayerNorm(axis=-1, create_scale=True,
                                 create_offset=True)(h)
                f = hk.Linear(hidden_size)(jax.nn.gelu(
                    hk.Linear(4 * hidden_size)(f)))
                h = h + hk.dropout(next(keys), dropout, f)
            else:
                h = hk.MultiHeadAttention(
                    num_heads=4, key_size=16, model_size=hidden_size,
                    w_init=w_init)(h, h, h, mask=mask)
                h = hk.LayerNorm(axis=-1, create_scale=True,
                                 create_offset=True)(h)
                h = hk.Linear(hidden_size)(h)
                h = hk.dropout(next(keys), dropout, h)
        return hk.Linear(horizon)(h), None
    return fwd


# ---------- Cox linear ----------


def _linear_forward(feature_dim, horizon):
    def fwd(x):
        beta = hk.get_parameter("beta", (feature_dim,), init=jnp.zeros)
        alpha = hk.get_parameter("alpha", (horizon,), init=jnp.zeros)
        # x: (B, T, D) -> (B, T); add per-horizon offset -> (B, T, H)
        scores = jnp.expand_dims(jnp.dot(x, beta), axis=-1)
        return scores + alpha, None
    return fwd


# ---------- GRU + temporal attention (Dynamic-DeepHit style) ----------


def _gru_attn_forward(hidden_size, horizon, feature_dim,
                      attention_hidden=None, causal=True, query_chunk=32):
    """GRU + temporal attention, Dynamic-DeepHit style.

    Faithful to the reference implementation (chl8856/Dynamic-DeepHit,
    ``class_DeepLongitudinal.py``): the RNN runs over the measurement
    history, attention scores are ``e_j = v^T tanh(W_h h_j + W_q x_query)``
    restricted to actually-measured steps, and the prediction consumes
    ``[current state, context]``.

    The reference makes ONE prediction, at the last measurement, so its
    attention over "all history" is causal by construction. TCSR instead
    needs ``h(k | x_t)`` at EVERY state, so the same mechanism is applied at
    each landmark t: query is the measurement at t, keys/values are hidden
    states j < t. With ``causal=False`` the previous behaviour is restored --
    a context pooled over the whole trajectory and broadcast to every
    position, which let a prediction at t=0 use the entire future.

    Query positions are processed in chunks of ``query_chunk``: the additive
    attention needs a (B, chunk, T, att) intermediate, which would be ~2 GB
    for a full T=363 sequence at batch 64.
    """
    att_hidden = attention_hidden or hidden_size

    def fwd(x):
        B, T, _ = x.shape
        h = jax.nn.relu(hk.Linear(hidden_size)(x))
        h = jnp.swapaxes(h, 0, 1)  # (T, B, H)
        core = hk.GRU(hidden_size)
        outputs, _ = hk.dynamic_unroll(core, h, core.initial_state(B))
        outputs = jnp.swapaxes(outputs, 0, 1)  # (B, T, H); causal in t

        measured = jnp.any(jnp.abs(x) > 0, axis=-1)  # (B, T) non-padding

        if not causal:
            last = jnp.repeat(outputs[:, -1:, :], T, axis=1)
            att_in = jnp.concatenate([outputs, last], axis=-1)
            att_scores = hk.Linear(1)(
                jax.nn.tanh(hk.Linear(att_hidden)(att_in))).squeeze(-1)
            tm = measured.at[:, -1].set(True)
            att_scores = jnp.where(tm, att_scores, -1e9)
            att = jax.nn.softmax(att_scores, axis=1)
            context = jnp.sum(att[:, :, None] * outputs, axis=1)
            context = jnp.repeat(context[:, None, :], T, axis=1)
        else:
            keys = hk.Linear(att_hidden)(outputs)   # W_h h_j
            query = hk.Linear(att_hidden)(x)        # W_q x_t (current measurement)
            score = hk.Linear(1)                    # shared across chunks
            idx = jnp.arange(T)
            parts = []
            for s0 in range(0, T, query_chunk):
                sl = slice(s0, min(s0 + query_chunk, T))
                e = score(jnp.tanh(keys[:, None, :, :]
                                   + query[:, sl][:, :, None, :])).squeeze(-1)
                allowed = (idx[None, :] < idx[sl][:, None])[None]  # j < t
                allowed = allowed & measured[:, None, :]
                e = jnp.where(allowed, e, -1e9)
                a = jax.nn.softmax(e, axis=-1)
                # t with no admissible history (t=0) gets a zero context
                a = jnp.where(jnp.any(allowed, axis=-1, keepdims=True), a, 0.0)
                parts.append(jnp.einsum("bct,bth->bch", a, outputs))
            context = jnp.concatenate(parts, axis=1)  # (B, T, H)

        fused = jnp.concatenate([outputs, context], axis=-1)
        hazard_logits = hk.Linear(horizon)(fused)
        cov_preds = hk.Linear(feature_dim)(outputs)
        return hazard_logits, cov_preds
    return fwd


# ---------- factory ----------


def build_backbone(name, horizon, feature_dim, seed, **kwargs):
    """Return an `hk.Transformed` for the requested backbone."""
    if name == "linear":
        fwd = _linear_forward(feature_dim, horizon)
    elif name == "tcn":
        fwd = _tcn_forward(
            num_channels=kwargs["num_channels"],
            kernel_size=kwargs.get("kernel_size", 3),
            dilation_factor=kwargs.get("dilation_factor", 2),
            horizon=horizon,
            seed=seed,
            dropout=kwargs.get("dropout", 0.2),
        )
    elif name == "transformer":
        fwd = _transformer_forward(
            hidden_size=kwargs["hidden_size"],
            seq_len=kwargs["seq_len"],
            horizon=horizon,
            num_layers=kwargs.get("num_layers", 3),
            dropout=kwargs.get("dropout", 0.2),
            seed=seed,
            residual=kwargs.get("residual", True),
        )
    elif name == "gru_attn":
        fwd = _gru_attn_forward(
            hidden_size=kwargs["hidden_size"],
            horizon=horizon,
            feature_dim=feature_dim,
            attention_hidden=kwargs.get("attention_hidden"),
            causal=kwargs.get("causal", True),
        )
    else:
        raise ValueError(f"Unknown backbone {name!r}")
    return hk.without_apply_rng(hk.transform(fwd))
