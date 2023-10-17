import os
import chex
import jax
import jax.numpy as jnp
import optax
from base_cox import BaseSA, ConfigParams, ModelState
from utils import train_test_split
from dataclasses import dataclass


@dataclass
class Config(ConfigParams):
    lambda_: float = 1.0
    num_steps: int = 5


class LambdaSA(BaseSA):

    def __init__(self, config_kwargs, seed, name="SurvivalAnalysis", **kwargs):
        super().__init__(config_kwargs, seed, name, **kwargs)

        new_config = Config.from_dict(config_kwargs)
        self.lambda_ = new_config.lambda_
        self.num_steps = new_config.num_steps

    def _targets(self, m, seqs, ts, cs):
        """Compute pseudo-targets and weights for the m-step backup."""
        H = self.config.horizon
        ys = jnp.zeros((len(seqs), H))
        ws = jnp.zeros((len(seqs), H))
        # Observed outcomes within first m steps.
        idx = (ts <= m) & ~cs  # Seqs that reached terminal state within window.
        ys = ys.at[idx, ts[idx] - 1].set(1.0)
        ws = ws.at[:, :m].set((jnp.arange(m) < ts[:, jnp.newaxis]).astype(float))
        # Predicted outcomes after first m steps.
        if m < H:
            # Seqs that are still active after the window.
            idx = (ts > m) | ((ts == m) & cs)
            nxt = seqs[idx, m]
            logits = self.forward(self.state.params, nxt)
            log_hs = jax.nn.log_sigmoid(logits)
            ys = ys.at[idx, m:].set(jnp.exp(log_hs[:, :-m]))
            ws = ws.at[idx, m].set(1.0)
            ws = ws.at[idx, (m + 1) :].set(jnp.exp(
                jnp.cumsum(
                    log_hs[:, : -(m + 1)] - logits[:, : -(m + 1)],
                    axis=1,
                )
            ))
        return (ys, ws)
    
    def _update_target(self, seqs, ts, cs, lambda_):
        # seqs = self.data['seqs']
        # ts = self.data['ts']
        # cs = self.data['cs']

        H = self.config.horizon
        cs = cs.astype(bool)
        n = len(seqs)

        # Exponentially decreasing multipliers.
        multipliers = lambda_ ** jnp.arange(H)
        multipliers = multipliers.at[:-1].set(multipliers[:-1] * 1 - lambda_)

        ys = jnp.zeros((len(seqs), H))
        ws = jnp.zeros((len(seqs), H))
        # Compute backup targets and weights at all steps.
        for m, mult in enumerate(multipliers, start=1):
            if mult == 0.0:
                # Multiplier is zero, we can ignore this step. This leads to
                # a significant speedup for `lambda_ = 0` and `lambda_ = 1`
                continue
            ys_m, ws_m = self._targets(m, seqs, ts, cs)
            ys += mult * ys_m
            ws += mult * ws_m

        return ys, ws
    
    def _inner_loop(self, seqs, ys, ws):
        """Training loop"""
        for epoch in range(self.config.num_epochs):

            self.state, loss = self.update(
                self.state,
                seqs,
                ys,
                ws
            )

            # log
            if epoch % self.config.log_interval == 0:
                print(f"Epoch: {epoch+1}/{self.config.num_epochs}")
                print(f"Train classification loss: {loss:.3f} at epoch {epoch}")
                print()

    def train(self):
        train_loss = []
        test_loss = []

        X_train, X_test, y_train, y_test, m_train, m_test,\
             ts_train, ts_test, cs_train, cs_test = train_test_split(self.data['seqs'],
                                                                    self.data['target'],
                                                                    self.data['mask'],
                                                                    self.data['ts'],
                                                                    self.data['cs'],
                                                                    rng=self._next_rng_key())
        for i in range(self.num_steps):
            print('\n\n Step (outer loop) {}\n\n'.format(i))
            ys, ws = self._update_target(X_train, ts_train, cs_train, self.lambda_)
            self._inner_loop(X_train[:, 0], ys, ws)
            
        if self.output_file is not None:
            if not os.path.exists(self.output_file):
                os.makedirs(self.output_file)