from functools import partial
import chex
import jax
import jax.numpy as jnp
import optax
from base_cox import BaseSA, ConfigParams, Params
from utils import LazyTimesDataGenerator, TimesDataGenerator, convert_to_jax_arrays, get_targets_and_masks, train_test_split
from dataclasses import dataclass
from tqdm import tqdm


@jax.jit
def bce_logits(targets, logits):
    """Compute the binary cross-entropy with logits.

    Letting `p = targets` and `q = sigmoid(logits)`, this function returns the
    binary cross-entropy `H(p, q) = -p * log(q) - (1 - p) * log(1 - q)`.
    """
    return -targets * logits + jax.nn.softplus(logits)


@dataclass
class Config(ConfigParams):
    lambda_: float = 1.0
    num_steps: int = 5
    target_lr: float = .0001

# Model state


@chex.dataclass(frozen=True)
class ModelState:
    """A structure of the current model state"""
    params: Params
    tgt_params: Params
    opt_state: optax.OptState


def _get_targets(b_tgt, h_tgt, lambda_, T, b_size):
    b_size = min(b_size, b_tgt.shape[0])
    stgt_init = jnp.roll(b_tgt[:, T-1], -1)
    stgt_init = stgt_init.at[:, T-1].set(0.0)
    htgt_init = jnp.zeros((b_size, T))
    carry_init = (stgt_init, htgt_init)

    def f(carry, h):
        next_b_tgt, next_h = carry
        h_tgt, b_tgt = h

        h_out = lambda_ * next_h + (1-lambda_) * next_b_tgt
        h_out = jnp.roll(h_out, 1)
        h_out = h_out.at[:, 0].set(h_tgt[:, 0])
        cond = h_tgt[:, 0] == 1
        h_out = jnp.where(cond[:, None], jnp.zeros((b_size, T)).at[:, 0].set(1), h_out)
        
        carry = (b_tgt, h_out)
        return carry, h_out

    h_aux = jnp.transpose(h_tgt, (1, 0, 2))
    s_aux = jnp.transpose(b_tgt, (1, 0, 2))
    _, out = jax.lax.scan(f, carry_init, (h_aux, s_aux), reverse=True)
    out = jnp.transpose(out, (1, 0, 2))
    return out


def _get_weights(s_wgt, h_wgt, h_tgt, c, lambda_, T, b_size):
    b_size = min(b_size, s_wgt.shape[0])
    w_init = s_wgt[:, T-1]
    w_init = jnp.roll(w_init, 1)
    w_init = w_init.at[:, 0].set(1.0)

    w_init = jnp.where(c[:, None], jnp.ones((b_size, T)), w_init)
    carry_init = (s_wgt[:, T-1], w_init)

    def f(carry, h):
        next_s_wgt, next_h = carry
        h_tgt, h_wgt, s_wgt = h

        h = lambda_ * next_h + (1-lambda_) * next_s_wgt
        aux = jnp.ones_like(h_wgt[:, 0], dtype=jnp.float32)
        value = jax.lax.select(c, aux,  h_wgt[:, 0].astype(jnp.float32))
        # value = jnp.where(c, aux, h_wgt[:, 0].astype(jnp.float32))
        h = jnp.roll(h, 1)
        h = h.at[:, 0].set(value)

        cond = h_tgt[:, 0] == 1
        h = jnp.where(cond[:, None], jnp.zeros((b_size, T)).at[:, 0].set(1), h)

        carry = (s_wgt, h)

        return carry, h

    y_aux = jnp.transpose(h_tgt, (1, 0, 2))
    h_aux = jnp.transpose(h_wgt, (1, 0, 2))
    s_aux = jnp.transpose(s_wgt, (1, 0, 2))

    _, out = jax.lax.scan(f, carry_init, (y_aux, h_aux, s_aux), reverse=True)

    out = jnp.transpose(out, (1, 0, 2))
    return out


class DeepLambdaSA(BaseSA):

    def __init__(self, config_kwargs, seed, name="DeepTCSR", **kwargs):
        super().__init__(config_kwargs, seed, name, **kwargs)

        config = Config.from_dict(config_kwargs)
        self.lambda_ = config.lambda_
        self.num_steps = config.num_steps

        # Online and Target updates
        tgt_update = partial(
            optax.incremental_update,
            step_size=config.target_lr
        )

        online_enc_update = self.online_enc_update

        # State of the model
        params, opt_state = self.state.values()

        # Online and Target initializations
        tgt_params = jax.tree_map(jnp.copy, params)

        # Update model state
        self.state = ModelState(
            params=params,
            tgt_params=tgt_params,
            opt_state=opt_state,
        )

        def get_targets(tgt_logits, ys):
            s_tgt = jax.nn.sigmoid(tgt_logits)
            h = _get_targets(s_tgt, ys, self.lambda_,
                             self.horizon, self.config.batch_size)
            return h

        self.get_targets = jax.jit(get_targets)

        def get_weights(s_ws, ys, cs, h_ws):
            w = _get_weights(s_ws, h_ws, ys, cs,
                             self.lambda_, self.horizon, self.config.batch_size)
            return w

        self.get_weights = jax.jit(get_weights)

        # Losses
        def loss_fn(onl_params, inputs, targets, ws, mask):
            logits = self.forward(onl_params, inputs)
            assert logits.shape == targets.shape
            loss = bce_logits(targets, logits)
            loss = jnp.mean(loss * ws * mask)
            return loss

        loss_fn = jax.value_and_grad(loss_fn, has_aux=False)

        # Update
        def update(model_state: ModelState,
                   inputs: chex.Array, targets: chex.Array, weights: chex.Array, mask: chex.Array):
            # Extract state
            onl_params, tgt_params, opt_state = model_state.values()

            # Compute loss
            loss, grad = loss_fn(
                onl_params,
                inputs,
                targets,
                weights,
                mask
            )
            # Update online encoder
            onl_params, opt_state = online_enc_update(
                onl_params,
                grad,
                opt_state
            )

            # Update target encoder
            tgt_params = tgt_update(onl_params, tgt_params)

            # Update model state
            model_state = model_state.replace(
                params=onl_params,
                tgt_params=tgt_params,
                opt_state=opt_state
            )

            return model_state, loss

        self.update = jax.jit(update)

    def get_train_test(self, test_size=.2):
        if self.config.calculate_tgt_and_mask:
            data_manager = TimesDataGenerator
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
                                                                                     test_size=test_size)
        subkey = self._next_rng_key()
        train_gen = data_manager(X=X_train, h_ws=hws_train,
                                 ts=ts_train, cs=cs_train,
                                 y=y_train, mask=m_train,
                                 batch_size=self.config.batch_size, rng=subkey)
        subkey = self._next_rng_key()
        test_gen = data_manager(X=X_test, h_ws=hws_test,
                                ts=ts_test, cs=cs_test,
                                y=y_test, mask=m_test,
                                batch_size=self.config.batch_size, rng=subkey)
        return train_gen, test_gen

    def train(self, train_gen=None, test_gen=None):

        if train_gen is None:
            train_gen, test_gen = self.get_train_test()

        losses = []
        iter_range = range(self.config.num_epochs)
        if self.config.verbose:
            iter_range = tqdm(iter_range)

        for epoch in iter_range:
            for batch in train_gen:
                if self.calculate_tgt_and_mask_at_epoch:
                    seqs, ts, cs = batch
                    
                    # hard target and weights calculation:
                    ys, h_ws, m = get_targets_and_masks(seqs, ts, cs, self.config.landmark)
                else:
                    seqs, _, cs, ys, m, h_ws = batch

                seqs, cs, ys, m, h_ws = convert_to_jax_arrays(
                    seqs, cs, ys, m, h_ws)

                # Get targets
                tgt_logits = self.forward(self.state.tgt_params, seqs)
                log_hs = jax.nn.log_sigmoid(tgt_logits)
                s_ws = jnp.exp(jnp.cumsum(log_hs - tgt_logits, axis=-1))
                s_ws = jnp.roll(s_ws, 1)
                s_ws = s_ws.at[:, :, 0].set(1.0)

                h = self.get_targets(tgt_logits, ys)
                ws = self.get_weights(s_ws, ys, cs, h_ws)

                self.state, loss = self.update(
                    self.state,
                    seqs,
                    h,
                    ws,
                    m
                )

                # log
                if epoch % self.config.log_interval == 0 and epoch > 1:
                    print(f"Epoch: {epoch+1}/{self.config.num_epochs}")
                    print(
                        f"Train classification loss: {loss.item():.3f} at epoch {epoch}")
                    print()

                losses.append(loss.item())
            train_gen.reset()

        # if self.output_file is not None:
        #     if not os.path.exists(self.output_file):
        #         os.makedirs(self.output_file)

        #     df = pd.DataFrame({
        #         "loss": losses,
        #     })
        #     path_csv = os.path.join(self.output_file, 'result.csv')
        #     df.to_csv(path_csv)

        return losses
