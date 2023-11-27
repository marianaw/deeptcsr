import haiku as hk
import jax
import jax.numpy as jnp
import optax
from typing import Any, Mapping, Text


def get_update_and_apply(optimizer):
    """ Get function that update the params and state of the optimizer"""

    def update_and_apply(params, grads, opt_state):
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state

    return update_and_apply


class Chomp1D(hk.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def __call__(self, x):
        return x[:, :, :-self.chomp_size]


class TemporalBlock(hk.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, seed, dropout=0.2):
        super().__init__()
        self.conv1 = hk.Conv1D(n_outputs, kernel_size,
                               stride=stride, padding=padding, rate=dilation, data_format='NC...')
        self.chomp1 = Chomp1D(padding[0])

        self.conv2 = hk.Conv1D(n_outputs, kernel_size,
                               stride=stride, padding=padding, rate=dilation, data_format='NC...')
        self.chomp2 = Chomp1D(padding[0])

        self.downsample = hk.Conv1D(
            n_outputs, 1, data_format='NC...') if n_inputs != n_outputs else None
        self.dropout = dropout
        self.keys_seq = hk.PRNGSequence(seed)

    def __call__(self, x):
        out = self.conv1(x)
        out = jax.nn.relu(self.chomp1(out))
        out = hk.dropout(next(self.keys_seq), self.dropout, out)

        out = self.conv2(out)
        out = jax.nn.relu(self.chomp2(out))
        out = hk.dropout(next(self.keys_seq), self.dropout, out)

        res = x if self.downsample is None else self.downsample(x)
        return jax.nn.relu(out + res)


class TCN(hk.Module):
    def __init__(self, num_inputs, num_channels, seed=33, kernel_size=2, stride=1, dilation_factor=2, dropout=0.2, **kwargs):
        super().__init__(**kwargs)

        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = dilation_factor ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            padding = (kernel_size-1) * dilation_size
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation_size,
                                     padding=(padding, padding), dropout=dropout, seed=seed)]
        self.layers = layers

    def __call__(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out


class MLP(hk.Module):
    """One hidden layer perceptron, with normalization."""

    def __init__(
        self,
        hidden_size: int,
        output_size: int,
        bn_config: Mapping[Text, Any],
        name: Text,
    ):
        super().__init__(name=name)
        self._hidden_size = hidden_size
        self._output_size = output_size
        self._bn_config = bn_config

        self.linear1 = hk.Linear(output_size=self._hidden_size, with_bias=True)
        self.norm = hk.BatchNorm(**self._bn_config)
        # self.norm = hk.LayerNorm(axis=-1,
        #             create_scale=True,
        #             create_offset=True)
        self.linear2 = hk.Linear(
            output_size=self._output_size, with_bias=False)

    def __call__(self, inputs: jnp.ndarray, is_training: bool) -> jnp.ndarray:
        out = self.linear1(inputs)
        out = self.norm(out, is_training=is_training)
        out = jax.nn.relu(out)
        out = self.linear2(out)
        return out


class CoxLinearModel(hk.Module):

    def __init__(self, n_feats, horizon, axis=1, name: str | None = None):
        super().__init__(name)
        self.horizon = horizon
        self.axis = axis
        self.beta = hk.get_parameter("beta", (n_feats,), init=jnp.zeros)
        self.alpha = hk.get_parameter("alpha", (horizon, ), init=jnp.zeros)

    def __call__(self, xs):
        return (
            jnp.expand_dims(jnp.dot(xs, self.beta), axis=self.axis)
            + self.alpha
        )


class TSTransformer(hk.Module):
    def __init__(self, hidden_size, seq_len, output_dim, dropout=0.2, num_layers=3, seed=42):
        super(TSTransformer, self).__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.output_dim = output_dim
        self.keys_seq = hk.PRNGSequence(seed)

    def __call__(self, x):
        x = hk.Linear(self.hidden_size)(x)
        mask = self._get_mask_future()
        w_init = hk.initializers.VarianceScaling(2 / self.num_layers)
        for _ in range(self.num_layers):
            x = hk.MultiHeadAttention(num_heads=4,
                                      key_size=16,
                                      model_size=self.hidden_size,
                                      w_init=w_init)(x, x, mask)
            x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
            x = hk.Linear(self.hidden_size)(x)
            x = hk.dropout(next(self.keys_seq), self.dropout, x)

        return x

    def _get_mask_future(self):
        n = self.seq_len
        mask = jnp.tril(jnp.ones((n, n), dtype=jnp.float32))
        return mask
