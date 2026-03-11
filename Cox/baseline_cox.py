from dataclasses import dataclass
import inspect
import chex
import optax

from tqdm import tqdm
from utils import LazyTimesDataGenerator, TgtMskDataGenerator, train_test_split
from .base_cox import BaseSA

Params = chex.ArrayTree
PRNGKey = chex.PRNGKey
State = chex.ArrayTree

# Config params


# @dataclass
# class ConfigParams:
#     """A structure for configuration"""
#     dataset_name: str
#     batch_size: int
#     learning_rate: float
#     log_interval: int
#     weight_decay: float
#     num_epochs: int
#     dataset_kwargs: dict
#     landmark: bool = False
#     output_file: str = None
#     early_stopping_patience: int = 5

#     @classmethod
#     def from_dict(cls, env):
#         """To ignore args that are not in the class,
#         see https://stackoverflow.com/questions/54678337/how-does-one-ignore-extra-arguments-passed-to-a-dataclass
#         """
#         return cls(**{
#             k: v for k, v in env.items()
#             if k in inspect.signature(cls).parameters
#         })


# Model state
@chex.dataclass(frozen=True)
class ModelState:
    """A structure of the current model state"""
    params: Params
    opt_state: optax.OptState


class SA(BaseSA):

    def get_train_test(self, test_size=.2, val_size=.2):
        if self.config.calculate_tgt_and_mask:
            data_manager = TgtMskDataGenerator
        else:
            data_manager = LazyTimesDataGenerator

        subkey = self._next_rng_key()
        X_train, X_test, y_train, y_test, hws_train, hws_test, \
            m_train, m_test, ts_train, ts_test, cs_train, cs_test = train_test_split(self.data['seqs'],
                                                                                     self.data['target'],
                                                                                     self.data['h_ws'],
                                                                                     self.data['mask'],
                                                                                     self.data['ts'],
                                                                                     self.data['cs'],
                                                                                     seed=self.seed,
                                                                                     test_size=test_size,
                                                                                     stratify=self.config.dataset_name == "mimic")
        X_val, y_val, hws_val, m_val, ts_val, cs_val = None, None, None, None, None, None
        if val_size is not None:
            X_val, X_test, y_val, y_test, hws_val, hws_test, \
                m_val, m_test, ts_val, ts_test, cs_val, cs_test = train_test_split(self.data['seqs'],
                                                                                   self.data['target'],
                                                                                   self.data['h_ws'],
                                                                                   self.data['mask'],
                                                                                   self.data['ts'],
                                                                                   self.data['cs'],
                                                                                   seed=self.seed,
                                                                                   test_size=val_size,
                                                                                   stratify=self.config.dataset_name == "mimic")
        subkey = self._next_rng_key()
        train_gen = data_manager(X=X_train,
                                 ts=ts_train, cs=cs_train,
                                 y=y_train, mask=m_train,
                                 batch_size=self.config.batch_size, rng=subkey)
        subkey = self._next_rng_key()
        test_gen = data_manager(X=X_test,
                                ts=ts_test, cs=cs_test,
                                y=y_test, mask=m_test,
                                batch_size=self.config.batch_size, rng=subkey)
        val_gen = None
        if X_val is not None:
            subkey = self._next_rng_key()
            val_gen = data_manager(X=X_val,
                                   ts=ts_val, cs=cs_val,
                                   y=y_val, mask=m_val,
                                   batch_siz=self.config.batch_size, rng=subkey)

        return train_gen, test_gen, val_gen

    def train(self, train_gen=None, test_gen=None, val_gen=None):
        """Training loop"""
        train_loss = []
        test_loss = []

        # For early-stopping:
        improvement_threshold = 1e-4
        best_val_loss = float('inf')
        patience_counter = 0

        if train_gen is None:
            train_gen, test_gen, val_gen = self.get_train_test()

        iter_range = range(self.config.num_epochs)
        if self.config.verbose:
            iter_range = tqdm(iter_range)

        for epoch in iter_range:
            tr_loss = self.train_step(train_gen)
            train_loss.append(tr_loss)

            if val_gen is not None:
                val_loss = self.test_step(val_gen)
                if val_loss < (best_val_loss - improvement_threshold):
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                # Check if early stopping should be triggered
                if patience_counter >= self.config.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch} (patience={self.config.early_stopping_patience})")
                    break

            if test_gen is not None:
                te_loss = self.test_step(test_gen)
                test_loss.append(te_loss)

        return train_loss, test_loss
