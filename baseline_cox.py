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

from networks import get_update_and_apply, HorizonBias
from utils import DataGenerator, batch_generator, get_data, kaplan_meier, train_test_split
from base_cox import BaseSA

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


# Model state
@chex.dataclass(frozen=True)
class ModelState:
    """A structure of the current model state"""
    params: Params
    opt_state: optax.OptState


class SA(BaseSA):
    
    def _get_train_test(self):
        subkey = self._next_rng_key()
        X_train, X_test, y_train, y_test, m_train, m_test,\
            = train_test_split(self.data['seqs'],
                               self.data['target'],
                               self.data['mask'],
                               rng=subkey)
        subkey = self._next_rng_key()
        train_gen = DataGenerator(X_train, y_train, m_train,
                                  self.config.batch_size, subkey)
        subkey = self._next_rng_key()
        test_gen = DataGenerator(X_test, y_test, m_test,
                                 self.config.batch_size, subkey)
        return train_gen, test_gen

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
                print(
                    f"Train classification loss: {tr_loss:.3f} at epoch {epoch}")
                print(
                    f"Test classification loss {te_loss:.3f} at epoch {epoch}")
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
