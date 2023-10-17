import haiku as hk
import jax
import jax.numpy as jnp
import optax
from typing import Any, Mapping, Optional, Text, Type


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
    self.linear2 = hk.Linear(output_size=self._output_size, with_bias=False)

  def __call__(self, inputs: jnp.ndarray, is_training: bool) -> jnp.ndarray:
    out = self.linear1(inputs)
    out = self.norm(out, is_training=is_training)
    out = jax.nn.relu(out)
    out = self.linear2(out)
    return out
  

class HorizonBias(hk.Module):

    def __init__(self, horizon, name: str | None = None):
        super().__init__(name)
        self.params = hk.get_parameter("alpha_t", (horizon,), init=jnp.zeros)

    def __call__(self, inputs):
        #If inputs is of shape (batch_size, dim)
        # if len(inputs.shape) == 1:
        #     # We add a dimension to account for the time step:
        #     # (batch_size, time_step, dim)
        #     # Observe that when calling this function dim = 1.
        #     inputs = jnp.expand_dims(inputs, axis=1)

        return inputs + self.params


class CoxLinearModel(hk.Module):

    def __init__(self, n_feats, horizon, name: str | None = None):
        super().__init__(name)
        self.horizon = horizon
        self.params = jnp.zeros(n_feats + horizon)

    def __call__(self, xs):   
        return (
            jnp.expand_dims(jnp.dot(xs, self.params[: -self.horizon]), axis=1)
            + self.params[-self.horizon :]
        )
