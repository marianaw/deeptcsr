from dataclasses import dataclass
from functools import partial
import inspect
import os
import pickle
from typing import Optional, Type
import chex
import jax
import jax.numpy as jnp
import haiku as hk
import optax

from networks import get_update_and_apply
from utils import get_data


Params = chex.ArrayTree
PRNGKey = chex.PRNGKey
State = chex.ArrayTree

#Config params
@dataclass
class ConfigParams:
    """A structure for configuration"""
    horizon: int
    lr: float
    path_data: str
    reduce: str = "sum"

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


class LinearCoxPH(hk.Module):

    def __init__(self, output_size: int, name: str | None = None):
        super().__init__(name)
        self.linear = hk.Linear(output_size)

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        return self.linear(inputs)


class SA:
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

        # Random key
        self._key = jax.random.PRNGKey(seed)

        # dataset info
        path_data = self.config.path_data
        seqs, target, mask = get_data(path_data, landmark=self.config.landmark)
        self.data = {'seqs': seqs,
                     'target': target,
                     'mask': mask}
        dim = seqs.shape[-1]
        H = self.config.horizon
        
        # Encoder
        def forward_fn(x):
            linear = LinearCoxPH(dim+H)
            return linear(x)

        _some_input = self.data['seqs'][0]
        _key = self._next_rng_key()
        forward = hk.without_apply_rng(hk.transform(forward_fn))
        params = forward.init(_key, _some_input, is_training=True)
        self.forward = forward.apply

        # Online encoder update
        optimizer = optax.adamw(learning_rate=self.config.learning_rate,
                                    weight_decay=self.config.weight_decay)
        opt_state = optimizer.init(params)
        online_enc_update = get_update_and_apply(optimizer)


        # State of the model
        self.state = ModelState(
            online_enc_params=params,
            opt_state=opt_state,
        )

        # Ouput
        self.output_file = self.config.output_file

        # Losses
        def bce_logits(targets, logits):
            return -targets * logits + jax.nn.softplus(logits)
        
        def loss_fn(params, inputs, targets):
            logits = self.forward(params, inputs)
            loss = optax.sigmoid_binary_cross_entropy(logits, targets)
            return loss

        loss_fn = jax.value_and_grad(loss_fn, has_aux=False)

        # Update
        def update(model_state: ModelState,
                   inputs: chex.Array, labels: chex.Array):
            # Extract state
            params, opt_state = model_state.values()

            # Compute loss
            loss, grad = loss_fn(
                params,
                inputs,
                labels
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

    def _next_rng_key(self) -> chex.PRNGKey:
        """Get the next rng subkey from class rngkey.
        Must *not* be called from under a jitted function!
        Returns:
            A fresh rng_key.
        """
        self._key, subkey = jax.random.split(self._key)
        return subkey

    def get_data(self, data_id):
        train_dl, test_dl = data_generator(self.path_data,
                                           data_id,
                                           self.dataset_config,
                                           self.config.batch_size)
        return train_dl, test_dl

    def train(self, data_id):
        """Training loop"""
        train_embdl = []
        train_classl = []
        test_classl = []
        train_accs = []
        test_accs = []

        train_dl, test_dl = self.get_data(data_id)

        for epoch in range(self.config.num_epochs):
            embdl, classifl, train_acc = self.train_step(
                train_dl, len(train_dl))
            test_classifl, test_acc = self.test_step(test_dl, len(test_dl))

            # store for plotting
            train_embdl.append(embdl)
            train_classl.append(classifl)
            train_accs.append(train_acc)
            test_classl.append(test_classifl)
            test_accs.append(test_acc)

            # log
            if epoch % self.config.log_interval == 0:
                print(f"Scenario {data_id}")
                print(f"Epoch: {epoch+1}/{self.config.num_epochs}")
                print(f"Train embedding loss: {embdl:.3f} at epoch {epoch}")
                print(
                    f"Train classification loss: {classifl:.3f} at epoch {epoch}")
                print(f"Train accuracy {train_acc:.3f} at epoch {epoch}")
                print(
                    f"Test classification loss {test_classifl:.3f} at epoch {epoch}")
                print(f"Test accuracy {test_acc:.3f} at epoch {epoch}")
                print()

        if self.output_file is not None:
            if not os.path.exists(self.output_file):
                os.makedirs(self.output_file)
            path_model = os.path.join(self.output_file, 'model.pt')
            path_state = os.path.join(self.output_file, 'state.pt')
            path_csv = os.path.join(self.output_file, 'result.csv')
            pickle.dump(self.state.online_enc_params, open(path_model, 'wb'))
            pickle.dump(self.state.online_enc_state, open(path_state, 'wb'))
            df = pd.DataFrame({
                "train_embd_loss": train_embdl,
                "train_classif_loss": train_classl,
                "test_classif_loss": test_classl,
                "train_acc": train_accs,
                "test_acc": test_accs,
            })
            df.to_csv(path_csv)

    def train_step(self, train_dl, size_loader=None):
        embdl = 0.0
        classifl = 0.0
        train_acc = 0.0

        size_loader = len(train_dl) if size_loader is None else size_loader
        for X, y in train_dl:
            X = jnp.array(X.numpy())
            y = jnp.array(y.numpy())
            key = self._next_rng_key()
            self.state, stats = self.update(
                self.state,
                key,
                X,
                y,
            )
            embdl += stats[1].item()
            classifl += stats[2].item()

            # Train accuracy
            out, _ = self.forward(
                params=self.state.online_enc_params,
                state=self.state.online_enc_state,
                is_training=False,
                x=X,
            )
            train_acc += multiclass_accuracy(
                y, out['y_pred']).item()

        embdl /= size_loader
        classifl /= size_loader
        train_acc /= size_loader

        return embdl, classifl, train_acc

    def test_step(self, test_dl, size_loader=None):
        # Test loss
        test_classifl = 0.0
        test_acc = 0.0

        size_loader = len(test_dl) if size_loader is None else size_loader
        for X, y in test_dl:
            X = jnp.array(X.numpy())
            y = jnp.array(y.numpy())

            # Get validation and test stats
            out, _ = self.forward(
                params=self.state.online_enc_params,
                state=self.state.online_enc_state,
                is_training=False,
                x=X,
            )

            test_classifl += optax.softmax_cross_entropy(out['y_pred'],
                                                         jax.nn.one_hot(y, self.dataset_config.num_classes)).mean().item()
            test_acc += multiclass_accuracy(y, out['y_pred']).item()

        test_classifl /= size_loader
        test_acc /= size_loader

        return test_classifl, test_acc
