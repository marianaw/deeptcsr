from functools import partial
import chex
import jax
import jax.numpy as jnp
import optax
from base_cox import BaseSA, ConfigParams, Params
from utils import TimesDataGenerator, train_test_split
from dataclasses import dataclass


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


def _get_targets(b_tgt, h_tgt, lambda_, T):
    stgt_init = jnp.insert(b_tgt[T-1, 1:], T, 0)
    htgt_init = jnp.zeros(T,)
    carry_init = (stgt_init, htgt_init)

    def f(carry, h):
        next_b_tgt, next_h = carry
        h_tgt, b_tgt = h

        h = lambda_ * next_h + (1-lambda_) * next_b_tgt
        h = jnp.insert(h[:-1], 0, h_tgt[0])
        h = jax.lax.select(h_tgt[0] == 1, jnp.zeros(T).at[0].set(1), h)

        carry = (b_tgt, h)
        return carry, h

    _, out = jax.lax.scan(f, carry_init, (h_tgt, b_tgt), reverse=True)
    return out


def _get_weights(s_wgt, h_wgt, h_tgt, c, lambda_, T):
    w_init = s_wgt[T-1]
    w_init = jnp.insert(w_init[:-1], 0, 1.0)
    w_init = jax.lax.select(c, jnp.ones(T), w_init)
    carry_init = (s_wgt[T-1], w_init)

    def f(carry, h):
        next_s_wgt, next_h = carry
        h_tgt, h_wgt, s_wgt = h

        h = lambda_ * next_h + (1-lambda_) * next_s_wgt
        value = jax.lax.select(c, 1.0,  h_wgt[0])
        h = jnp.insert(h[:-1], 0, value)
        h = jax.lax.select(h_tgt[0] == 1, jnp.zeros(T).at[0].set(1), h)

        carry = (s_wgt, h)
        return carry, h

    _, out = jax.lax.scan(f, carry_init, (h_tgt, h_wgt, s_wgt), reverse=True)
    return out


def get_mapped_f_factory(f, lambda_, T, in_axis):
    single_f = partial(f, lambda_=lambda_, T=T)
    batched_f = jax.vmap(single_f, in_axes=in_axis)
    return batched_f


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

        # Function to get the targets
        get_tgt = get_mapped_f_factory(
                _get_targets, self.lambda_, self.horizon, in_axis=(0, 0))
        
        def get_targets(tgt_logits, ys):
            s_tgt = jax.nn.sigmoid(tgt_logits)
            h = get_tgt(s_tgt, ys)
            return h

        self.get_targets = jax.jit(get_targets)

        # Function to get weights
        get_ws = get_mapped_f_factory(
                _get_weights, self.lambda_, self.horizon, in_axis=(0, 0, 0, 0))
        
        def get_weights(s_ws, ys, cs, h_ws):
            w = get_ws(s_ws, h_ws, ys, cs)
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
        subkey = self._next_rng_key()
        X_train, X_test, y_train, y_test, hws_train, hws_test,\
            m_train, m_test, ts_train, ts_test, cs_train, cs_test = train_test_split(self.data['seqs'],
                                                                                     self.data['target'],
                                                                                     self.data['h_ws'],
                                                                                     self.data['mask'],
                                                                                     self.data['ts'],
                                                                                     self.data['cs'],
                                                                                     rng=subkey,
                                                                                     test_size=test_size)
        subkey = self._next_rng_key()
        train_gen = TimesDataGenerator(X=X_train, h_ws=hws_train,
                                       ts=ts_train, cs=cs_train,
                                       y=y_train, mask=m_train,
                                       batch_size=self.config.batch_size, rng=subkey)
        subkey = self._next_rng_key()
        test_gen = TimesDataGenerator(X=X_test, h_ws=hws_test,
                                      ts=ts_test, cs=cs_test,
                                      y=y_test, mask=m_test,
                                      batch_size=self.config.batch_size, rng=subkey)
        return train_gen, test_gen

    def train(self, train_gen=None):

        if train_gen is None:
            train_gen, _ = self.get_train_test()

        losses = []
        for epoch in range(self.config.num_epochs):
            for seqs, ts, cs, ys, m, h_ws in train_gen:

                # Get targets
                tgt_logits = self.forward(self.state.tgt_params, seqs)
                log_hs = jax.nn.log_sigmoid(tgt_logits)
                tgt_surv = jnp.exp(jnp.cumsum(log_hs - tgt_logits, axis=1))
                s_ws = jnp.insert(tgt_surv, 0, 1.0, axis=-1)
                s_ws = s_ws[:, :, :-1]

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
