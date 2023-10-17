from dataclasses import dataclass
import inspect
import os
import pickle
import chex
import jax
import jax.numpy as jnp
import haiku as hk
import optax
import pandas as pd

from networks import CoxLinearModel, get_update_and_apply, HorizonBias
from utils import DataGenerator, batch_generator, get_data, kaplan_meier, train_test_split


Params = chex.ArrayTree
PRNGKey = chex.PRNGKey
State = chex.ArrayTree

#Config params
@dataclass
class ConfigParams:
    """A structure for configuration"""
    horizon: int
    path_data: str
    dataset_name: str
    batch_size: int
    learning_rate: float
    log_interval: int
    weight_decay: float
    num_epochs: int
    landmark: bool = False
    output_file: str = None

    @classmethod
    def from_dict(cls, env):    
        """To ignore args that are not in the class,
        see https://stackoverflow.com/questions/54678337/how-does-one-ignore-extra-arguments-passed-to-a-dataclass
        """  
        return cls(**{
            k: v for k, v in env.items() 
            if k in inspect.signature(cls).parameters
        })


#Model state
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
        H = self.config.horizon

        # Random key
        self._key = jax.random.PRNGKey(seed)

        # dataset info
        path_data = self.config.path_data
        seqs, target, mask, ts, cs = get_data(path_data, landmark=self.config.landmark)
        n = seqs.shape[0]
        seqs = jnp.concatenate((seqs, jnp.tile(jnp.eye(H), (n, 1, 1))), axis=-1)
        self.data = {'seqs': seqs,
                     'target': target,
                     'mask': mask,
                     'ts': ts,
                     'cs': cs}
        dim = seqs.shape[-1]
        
        # Encoder
        def forward_fn(x):
            cox = CoxLinearModel(dim, H)
            return cox(x)

        _some_input = self.data['seqs'][:20]
        # _some_input = _some_input.reshape(1, *_some_input.shape)
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

    def _get_train_test(self):
        subkey = self._next_rng_key()
        X_train, X_test, y_train, y_test, m_train, m_test,\
             ts_train, ts_test, cs_train, cs_test = train_test_split(self.data['seqs'],
                                                                             self.data['target'],
                                                                             self.data['mask'],
                                                                             rng=subkey)
        subkey = self._next_rng_key()
        train_gen = DataGenerator(X_train, y_train, m_train,
                                  ts_train, cs_train, self.config.batch_size, subkey)
        subkey = self._next_rng_key()
        test_gen = DataGenerator(X_test, y_test, m_test, 
                                 ts_test, cs_test, self.config.batch_size, subkey)
        return train_gen, test_gen

    def _next_rng_key(self) -> chex.PRNGKey:
        """Get the next rng subkey from class rngkey.
        Must *not* be called from under a jitted function!
        Returns:
            A fresh rng_key.
        """
        self._key, subkey = jax.random.split(self._key)
        return subkey

    def train(self):
        """Training loop"""
        train_loss = []
        test_loss = []

        train_gen, test_gen = self._get_train_test()
        for epoch in range(self.config.num_epochs):
            tr_loss = self.train_step(train_gen)
            te_loss = self.test_step(test_gen)
            train_loss.append(tr_loss)
            test_loss.append(te_loss)

            # log
            if epoch % self.config.log_interval == 0:
                print(f"Epoch: {epoch+1}/{self.config.num_epochs}")
                print(f"Train classification loss: {tr_loss:.3f} at epoch {epoch}")
                print(f"Test classification loss {te_loss:.3f} at epoch {epoch}")
                print()

        if self.output_file is not None:
            if not os.path.exists(self.output_file):
                os.makedirs(self.output_file)
            path_model = os.path.join(self.output_file, 'model.pt')
            path_state = os.path.join(self.output_file, 'state.pt')
            path_csv = os.path.join(self.output_file, 'result.csv')
            pickle.dump(self.state.params, open(path_model, 'wb'))
            pickle.dump(self.state.opt_state, open(path_state, 'wb'))
            df = pd.DataFrame({
                "train_classif_loss": train_loss,
                "test_classif_loss": test_loss,
            })
            df.to_csv(path_csv)

    def train_step(self, train_gen):
        epoch_loss = 0.0
        count = 0

        for X, y, m in train_gen:
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
        logits = self.forward(self.state.params, xs).squeeze() # We call this for the first state.
        log_hs = jax.nn.log_sigmoid(logits)
        surv = jnp.exp(jnp.cumsum(log_hs - logits, axis=1))
        return jnp.insert(surv, 0, 1.0, axis=1)
    
    def integrated_brier_score(self, xs, ts, cs):
        """Compute the integrated Brier score."""
        cs = cs.astype(jnp.bool_)
        surv = self.survival_curve(xs)
        ws = kaplan_meier(ts - ~cs, ~cs)
        t_max = jnp.max(ts)
        tot = 0.0
        for h in range(1, t_max + 1):
            # Sequences that terminated.
            idx = (ts <= h) & ~cs
            tot += jnp.sum((1 / ws[ts[idx] - 1]) * (0.0 - surv[idx, h - 1]) ** 2)
            # Sequences that are still active.
            idx = (ts > h) | ((ts == h) & cs)
            tot += jnp.sum((1 / ws[h - 1]) * (1.0 - surv[idx, h - 1]) ** 2)
        return tot / (t_max * len(ts))
