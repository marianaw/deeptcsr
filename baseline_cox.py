from dataclasses import dataclass
import inspect
import os
import pickle
import chex
import optax
import pandas as pd

from utils import TgtMskDataGenerator, train_test_split
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
    
    def get_train_test(self, test_size=.2):
        subkey = self._next_rng_key()
        X_train, X_test, y_train, y_test, m_train, m_test, _, _,\
           ts_train, ts_test, cs_train, cs_test = train_test_split(self.data['seqs'],
                               self.data['target'],
                               self.data['h_ws'],
                               self.data['mask'],
                               self.data['ts'],
                               self.data['cs'],
                               rng=subkey,
                               test_size=test_size)
        subkey = self._next_rng_key()
        train_gen = TgtMskDataGenerator(X=X_train,
                                  ts=ts_train, cs=cs_train, 
                                  y=y_train, mask=m_train,
                                  batch_size=self.config.batch_size, rng=subkey)
        subkey = self._next_rng_key()
        test_gen = TgtMskDataGenerator(X=X_test, 
                                 ts=ts_test, cs=cs_test,
                                 y=y_test, mask=m_test,
                                 batch_size=self.config.batch_size, rng=subkey)
        return train_gen, test_gen

    def train(self, train_gen=None, test_gen=None):
        """Training loop"""
        train_loss = []
        test_loss = []

        if train_gen is None or test_gen is None:
            train_gen, test_gen = self.get_train_test()

        for epoch in range(self.config.num_epochs):
            tr_loss = self.train_step(train_gen)
            te_loss = self.test_step(test_gen)
            train_loss.append(tr_loss)
            test_loss.append(te_loss)

            # log
            # if epoch % self.config.log_interval == 0:
            #     print(f"Epoch: {epoch+1}/{self.config.num_epochs}")
            #     print(
            #         f"Train classification loss: {tr_loss:.3f} at epoch {epoch}")
            #     print(
            #         f"Test classification loss {te_loss:.3f} at epoch {epoch}")
            #     print()

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
        
        return train_loss, test_loss
