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

class CustomBatchNorm(hk.Module):
    def __init__(
        self,
        create_scale: bool,
        create_offset: bool,
        eps: float = 1e-5,
    ):

        super().__init__()
        self.create_scale = create_scale
        self.create_offset = create_offset
        self.eps = eps
        self.scale_init = jnp.ones
        self.offset_init = jnp.zeros
        self.channel_index = hk.get_channel_index("channels_last")

    def __call__(
        self,
        inputs: jax.Array,
    ) -> jax.Array:
        """Computes the normalized version of the input.

        Args:
        inputs: An array, where the data format is ``[..., C]``.
        Returns:
        The array, normalized across all but the last dimension.
        """

        channel_index = self.channel_index
        if channel_index < 0:
            channel_index += inputs.ndim

        axis = [i for i in range(inputs.ndim) if i != channel_index]

        mean = jnp.mean(inputs, axis, keepdims=True)
        mean_of_squares = jnp.mean(jnp.square(inputs), axis, keepdims=True)
        var = mean_of_squares - jnp.square(mean)

        w_shape = [1 if i in axis else inputs.shape[i]
                   for i in range(inputs.ndim)]
        w_dtype = inputs.dtype

        if self.create_scale:
            scale = hk.get_parameter(
                "scale", w_shape, w_dtype, self.scale_init)
        else:
            scale = jnp.ones([], dtype=w_dtype)

        if self.create_offset:
            offset = hk.get_parameter(
                "offset", w_shape, w_dtype, self.offset_init)
        else:
            offset = jnp.zeros([], dtype=w_dtype)

        eps = jax.lax.convert_element_type(self.eps, var.dtype)
        inv = scale * jax.lax.rsqrt(var + eps)
        return (inputs - mean) * inv + offset
    
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
    self.linear2 = hk.Linear(output_size=self._output_size, with_bias=False)

  def __call__(self, inputs: jnp.ndarray, is_training: bool) -> jnp.ndarray:
    out = self.linear1(inputs)
    out = self.norm(out, is_training=is_training)
    out = jax.nn.relu(out)
    out = self.linear2(out)
    return out
