import sys
sys.path.append('../')

from functools import partial
from dataclasses import dataclass
import inspect
import json
import os
import pickle
import chex
import jax
import jax.numpy as jnp
import haiku as hk
import optax
import numpy as np

from networks import TCN, CoxLinearModel, TSTransformer, get_update_and_apply
from utils import concordance_index, convert_to_jax_arrays, get_data, get_targets_and_masks, kaplan_meier, load_preprocessed_dataset


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
    verbose: bool
    calculate_tgt_and_mask: bool = True
    landmark: bool = False
    output_file: str = None
    ckpt_path: str = None
    stratify: bool = None
    early_stopping_patience: int = 5
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
        self.calculate_tgt_and_mask_at_epoch = not self.config.calculate_tgt_and_mask

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
                                                       self.config.calculate_tgt_and_mask,
                                                       self.config.dataset_kwargs)
            seqs = seqs.astype(np.float32)

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
            raise Exception(
                'Backbone not understood, should be transfomer, linear, or tcn.')

        _some_input = jnp.array(self.data['seqs'][:2])
        _key = self._next_rng_key()
        forward = hk.without_apply_rng(hk.transform(forward_fn))
        params = forward.init(_key, _some_input)
        if self.config.ckpt_path is not None:
            params_path = os.path.join(self.config.ckpt_path.format(seed), 'model.pt')
            params = pickle.load(open(params_path, 'rb'))
        self.forward = forward.apply

        # Online encoder update
        optimizer = optax.adamw(learning_rate=self.config.learning_rate,
                                weight_decay=self.config.weight_decay)
        opt_state = optimizer.init(params)
        if self.config.ckpt_path is not None:
            state_path = os.path.join(self.config.ckpt_path.format(seed), 'state.pt')
            opt_state = pickle.load(open(state_path, 'rb'))
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
        def median_time_idx(surv):
            # k = first index where S <= q; if never crosses, use T
            T = surv.shape[0]
            crosses = surv <= q
            has = jnp.any(crosses)
            k = jnp.where(has, jnp.argmax(crosses), T).astype(jnp.int32)
            return k.astype(jnp.float32)

        S = self.survival_curve(x)  # shape (N, T) or (N, ?, T)
        if S.ndim == 2:
            return jax.vmap(median_time_idx)(S)
        else:
            return jax.vmap(median_time_idx)(S[:, 0])

    def train_step(self, train_gen):
        epoch_loss = 0.0
        count = 0

        for batch in train_gen:
            if self.calculate_tgt_and_mask_at_epoch:
                X, ts, cs = batch

                # hard target and weights calculation:
                y, m, _ = get_targets_and_masks(
                    X, ts, cs, self.config.landmark)
            else:
                X, y, m = batch

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
        for batch in test_gen:
            if self.calculate_tgt_and_mask_at_epoch:
                X, ts, cs = batch

                # hard target and weights calculation:
                y, m, _ = get_targets_and_masks(
                    X, ts, cs, self.config.landmark)
            else:
                X, y, m = batch

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

    @property
    def output_path(self):
        if self.config.output_file is not None:
            path = self.config.output_file
            if self.config.dataset_name == 'single_task':
                task_id = self.config.dataset_kwargs['task_id']
                path = os.path.join(path, f'taskid_{task_id}')

            ext = f'seed_{self.seed}'
            path = os.path.join(path, ext)            
            if not os.path.exists(path):
                os.makedirs(path)
        else:
            path = None

        return path

    def eval(self, test_gen, suffix=''):
        seqs = test_gen.X
        ts = test_gen.ts
        cs = test_gen.cs

        surv = self.survival_curve(seqs)
        bs = self.integrated_brier_score(surv[:, 0], ts, cs)
        scores = self.scores(seqs, q=0.5)
        ci = concordance_index(scores, ts, cs)

        output_path = self.output_path
        if output_path is not None:
            ci = ci.item()
            bs = bs.item()
            path_result = os.path.join(output_path, f'results_{suffix}.json')
            data = {'ci': ci, 'bs': bs}
            with open(path_result, 'w') as json_file:
                json.dump(data, json_file)

        return bs, ci

    def save(self):
        output_path = self.output_path
        if output_path is not None:
            path_model = os.path.join(output_path, 'model.pt')
            path_state = os.path.join(output_path, 'state.pt')
            pickle.dump(self.state.params, open(path_model, 'wb'))
            pickle.dump(self.state.opt_state, open(path_state, 'wb'))
