from functools import partial
from dataclasses import dataclass
import inspect
import os
import pickle
import chex
import jax
import jax.numpy as jnp
import haiku as hk
import optax

from networks import TCN, CoxLinearModel, TSTransformer, get_update_and_apply
from utils import convert_to_jax_arrays, get_data, kaplan_meier, load_preprocessed_dataset


Params = chex.ArrayTree
PRNGKey = chex.PRNGKey
State = chex.ArrayTree

# Config params


@dataclass
class ConfigParams:
    """A structure for configuration"""
    dataset_name: str
    batch_size: int
    learning_rate: float
    log_interval: int
    weight_decay: float
    num_epochs: int
    dataset_kwargs: dict
    axis: int
    arch: dict
    preprocessed_data: bool
    landmark: bool = False
    output_file: str = None
    # horizon: int

    @classmethod
    def from_dict(cls, env):
        """To ignore args that are not in the class,
        see https://stackoverflow.com/questions/54678337/how-does-one-ignore-extra-arguments-passed-to-a-dataclass
        """
        return cls(**{
            k: v for k, v in env.items()
            if k in inspect.signature(cls).parameters
        })


# Model state
@chex.dataclass(frozen=True)
class ModelState:
    """A structure of the current model state"""
    params: Params
    opt_state: optax.OptState


class BaseSA:
    def __init__(
        self,
        config_kwargs,
        seed,
        name="SurvivalAnalysis",
        **kwargs,
    ):
        # Name
        self.name = name

        # Config
        self.config = ConfigParams.from_dict(config_kwargs)
        H = self.config.dataset_kwargs['horizon']
        self.horizon = H

        # Random key
        self.seed = seed
        self._key = jax.random.PRNGKey(seed)

        # dataset info
        if self.config.preprocessed_data:
            data_path = self.config.dataset_kwargs['data_path']
            seqs, ts, cs, h_tgt, h_ws, mask = load_preprocessed_dataset(
                data_path)
        else:
            seqs, ts, cs, h_tgt, h_ws, mask = get_data(self.config.dataset_name,
                                                       self.config.landmark,
                                                       self.config.dataset_kwargs)
        self.data = {'seqs': seqs,
                     'ts': ts,
                     'cs': cs,
                     'h_ws': h_ws,
                     'target': h_tgt,
                     'mask': mask}

        # Define architecture
        arch_type = self.config.arch['type']
        arch_kwargs = self.config.arch['arch_kwargs']
        arch_kwargs['seed'] = seed
        if arch_type == 'tcn':
            dim = arch_kwargs['num_channels'][-1]

            def forward_fn(x):
                tcn = TCN(**arch_kwargs)
                # (batch, channels, H)
                perm_x = jnp.transpose(x, axes=(0, 2, 1))
                out = tcn(perm_x)
                # (batch, H, channels)
                out = jnp.transpose(out, axes=(0, 2, 1))
                out = hk.Linear(self.horizon)(out)
                return out
        
        elif arch_type == 'transformer':
            def forward_fn(x):
                ts_transformer = TSTransformer(**arch_kwargs)
                x = ts_transformer(x)
                x = hk.Linear(self.horizon)(x)
                return x 

        elif arch_type == 'linear':
            dim = seqs.shape[-1]

            # Encoder
            def forward_fn(x):
                cox = CoxLinearModel(dim, H, axis=self.config.axis)
                out = cox(x)
                return out
            
        else:
            raise Exception('Backbone not understood, should be transfomer, linear, or tcn.')

        _some_input = jnp.array(self.data['seqs'][:2])
        _key = self._next_rng_key()
        forward = hk.without_apply_rng(hk.transform(forward_fn))
        params = forward.init(_key, _some_input)
        self.forward = forward.apply

        # Online encoder update
        optimizer = optax.adamw(learning_rate=self.config.learning_rate,
                                weight_decay=self.config.weight_decay)
        opt_state = optimizer.init(params)
        online_enc_update = get_update_and_apply(optimizer)

        # State of the model
        self.state = ModelState(
            params=params,
            opt_state=opt_state,
        )

        # Ouput
        self.output_file = self.config.output_file

        # Losses
        def loss_fn(params, inputs, targets, ws):
            logits = self.forward(params, inputs)
            loss = optax.sigmoid_binary_cross_entropy(logits, targets)
            loss = jnp.mean(loss * ws)
            return loss

        loss_fn = jax.value_and_grad(loss_fn, has_aux=False)

        # Update
        def update(model_state: ModelState,
                   inputs: chex.Array, labels: chex.Array, mask: chex.Array):
            # Extract state
            params, opt_state = model_state.values()

            # Compute loss
            loss, grad = loss_fn(
                params,
                inputs,
                labels,
                mask
            )
            # Update online encoder
            params, opt_state = online_enc_update(
                params,
                grad,
                opt_state
            )

            # Update model state
            model_state = model_state.replace(
                params=params,
                opt_state=opt_state
            )

            return model_state, loss

        self.update = jax.jit(update)
        self.online_enc_update = online_enc_update

    def _next_rng_key(self) -> chex.PRNGKey:
        """Get the next rng subkey from class rngkey.
        Must *not* be called from under a jitted function!
        Returns:
            A fresh rng_key.
        """
        self._key, subkey = jax.random.split(self._key)
        return subkey

    # Scores
    def scores(self, x, q=0.5):

        def median_fn(surv):
            mask = jnp.where(surv > q, 1, 0)
            idx = jnp.maximum(0, mask.sum()-1)
            return surv[idx]

        median = jax.vmap(median_fn)
        surv = self.survival_curve(x)
        if len(surv.shape) == 2:
            scores = median(surv)
        else:
            scores = median(surv[:,0])
        return scores
        # backbone_params = {
        #     key: value for key, value in self.state.params.items() if 'tcn_scores' in key}
        # beta = self.state.params['cox_linear_model']['beta']

        # out = self.backbone(backbone_params, x)
        # return -jnp.dot(out, beta)

    def train_step(self, train_gen):
        epoch_loss = 0.0
        count = 0

        for X, y, m in train_gen:

            X, y, m = convert_to_jax_arrays(X, y, m)
            
            self.state, loss = self.update(
                self.state,
                X,
                y,
                m
            )
            epoch_loss += loss.item()
            count += 1

        epoch_loss /= count
        train_gen.reset()
        return epoch_loss

    def test_step(self, test_gen):
        # Test loss
        epoch_loss = 0.0
        count = 0
        for X, y, m in test_gen:
            X, y, m = convert_to_jax_arrays(X, y, m)

            # Get validation and test stats
            out = self.forward(
                params=self.state.params,
                x=X,
            )
            loss = optax.sigmoid_binary_cross_entropy(out, y)
            epoch_loss += (loss * m).mean().item()
            count += 1

        epoch_loss /= count
        test_gen.reset()
        return epoch_loss

    def survival_curve(self, xs):
        """Compute the fixed-horizon survival CCDF, a.k.a. survival curve.

        Letting `S(k | x)` be the survival at step `k` from state `x`, this
        function returns

            [ 1  S(1 | x)  ...  S(K | x) ]

        where `K` is the horizon.
        """
        logits = self.forward(self.state.params, xs)
        if self.config.axis == 1:
            logits = logits.squeeze()  # We call this for the first state.
        log_hs = jax.nn.log_sigmoid(logits)
        surv = jnp.exp(jnp.cumsum(log_hs - logits, axis=self.config.axis))
        return jnp.insert(surv, 0, 1.0, axis=self.config.axis)

    def brier_score(self, h, surv, ts, cs):
        cs = cs.astype(jnp.bool_)
        ws = kaplan_meier(ts - ~cs, ~cs)

        # Sequences that terminated.
        mask = jnp.where((ts <= h) & ~cs, 1, 0)
        aux = (1/ws[ts-1]) * (0.0 - surv[:, h-1])**2 * mask
        aux = jnp.where(jnp.isinf(aux), 0, aux)
        aux = jnp.where(jnp.isnan(aux), 0, aux)
        tot = jnp.sum(aux)

        # Sequences that are still active.
        mask = jnp.where((ts > h) | ((ts == h) & cs), 1, 0)
        aux = (1 / ws[h - 1]) * (1.0 - surv[:, h - 1]) ** 2 * mask
        aux = jnp.where(jnp.isinf(aux), 0, aux)
        aux = jnp.where(jnp.isnan(aux), 0, aux)
        aux = jnp.sum(aux)
        tot += aux
        return tot

    # @jax.jit
    def integrated_brier_score(self, surv, ts, cs):
        brier = partial(self.brier_score, surv=surv, ts=ts, cs=cs)
        f = jax.vmap(brier)
        t_max = jnp.max(ts)
        hs = jnp.arange(1, t_max+1)
        return jnp.sum(f(hs) / (t_max * len(ts)))

    def old_integrated_brier_score(self, xs, ts, cs):
        """Compute the integrated Brier score."""
        cs = cs.astype(jnp.bool_)
        surv = self.survival_curve(xs)
        ws = kaplan_meier(ts - ~cs, ~cs)
        t_max = jnp.max(ts)
        tot = 0.0
        for h in range(1, t_max + 1):
            # Sequences that terminated.
            idx = (ts <= h) & ~cs
            tot += jnp.sum((1 / ws[ts[idx] - 1]) *
                           (0.0 - surv[idx, h - 1]) ** 2)
            # Sequences that are still active.
            idx = (ts > h) | ((ts == h) & cs)
            tot += jnp.sum((1 / ws[h - 1]) * (1.0 - surv[idx, h - 1]) ** 2)
        return tot / (t_max * len(ts))

    def save(self):
        if self.config.output_file is not None:
            ext = f'seed_{self.seed}'
            if self.config.dataset_name == 'single_task':
                task_id = self.config.dataset_kwargs['task_id']
                ext = ext + f'_taskid_{task_id}'

            output_path = os.path.join(self.config.output_file,
                                       self.config.dataset_name,
                                       ext)

            if not os.path.exists(output_path):
                os.makedirs(output_path)
            path_model = os.path.join(output_path, 'model.pt')
            path_state = os.path.join(output_path, 'state.pt')
            pickle.dump(self.state.params, open(path_model, 'wb'))
            pickle.dump(self.state.opt_state, open(path_state, 'wb'))
