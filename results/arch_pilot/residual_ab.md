# Transformer residual-connection A/B

`_transformer_forward` originally had **no residual connections**: each layer
replaced the hidden state (`h = attn(h)` -> LayerNorm -> Linear -> dropout)
with no skip path. Standard pre-norm blocks were added behind a `residual`
flag (default True; `residual=false` reproduces the old behaviour).

Setting: tc_cox, hidden=64, layers=2, lambda=0.1, tau=0.05, seed 0,
400-epoch cap + early stopping, identical splits.

| dataset | residual | test CI | test IBS |
|---------|----------|---------|----------|
| nasa    | **true** | **0.6355** | 0.4981 |
| nasa    | false    | 0.5513  | 0.4923 |
| scania  | **true** | 0.5000  | 0.0908 |
| scania  | false    | 0.4998  | 0.0890 |

**Conclusion.** Residuals give a large gain on NASA (+0.084 C-index) and are
neutral on Scania, so `residual=true` is used for the final sweep.

**Open issue (not fixed here).** The Cox/transformer family sits at chance on
Scania (CI ~= 0.50) for *every* width, depth, and residual setting tried. A
plausible substantive explanation: the Cox arms predict from the initial
state only (`axis=2` -> `logits[:, 0, :]`), and a truck's first readout
carries near-zero cumulative wear, so it genuinely may not discriminate
failure ~100 steps later. Note the DDH backbone is NOT causal at position 0 --
its attention context spans the whole sequence (`last = outputs[:, -1:, :]`
broadcast to all positions), so its high Scania scores (CI 0.89-0.95) use
information the Cox arms cannot see. Worth resolving before publishing any
Cox-vs-DDH comparison on Scania.

**Caveat.** The capacity ranking in `arch_pilot.md` was measured *without*
residuals, where depth actively hurt. With residuals, deeper configurations
may now be viable; h64/l2 was the best non-residual config and is carried
forward, but the depth question was not re-pilots for time/budget reasons.

---

## Follow-up: causal-mask bug in the transformer (found after the A/B)

A leakage probe (perturb inputs only at t>=6, check whether outputs at t<6
change) showed the transformer's t=0 output depending on future timesteps,
which is impossible if the causal mask worked. Cause:

```python
hk.MultiHeadAttention(...)(a, a, mask)      # WRONG
```

`hk.MultiHeadAttention.__call__` is `(query, key, value, mask=None)`, so the
lower-triangular `mask` was being consumed as **value** and no mask was ever
applied. The attention output was therefore built from a constant triangular
matrix rather than from the input representations, and the model was not
causal. Fixed to:

```python
hk.MultiHeadAttention(...)(a, a, a, mask=mask)   # mask now (1,1,T,T)
```

Re-probed after the fix: perturbing t>=6 leaves t=0..5 bit-identical
(max|delta| = 0) and changes t>=6. This bug predates the residual work and
affected every transformer/Cox run ever produced in this project, including
the legacy results. The 12 sweep runs completed before the fix were deleted
and the sweep restarted.

## Open: the DDH backbone is not causal either

The same probe shows `gru_attn` output at t=0 changing when only t>=6 is
perturbed. This is by construction, not a coding slip: the attention context
is pooled over all T steps (`last = outputs[:, -1:, :]` broadcast to every
position, attention masked only for padding), so a prediction made at state 0
uses the whole trajectory. Since the Cox arms are evaluated from state 0
(`axis=2` -> `logits[:, 0, :]`), DDH-vs-Cox comparisons are not like-for-like.
Inc-TCSR vs D-TCSR within the DDH family remains internally fair (both arms
share the backbone). Needs a decision before publication.

## Resolved: DDH attention made causal, per the reference implementation

Checked against the reference code (chl8856/Dynamic-DeepHit,
`class_DeepLongitudinal.py`). The real Dynamic-DeepHit:

* splits each subject into `x_hist` (measurements 1..M-1) and `x_last`
  (the current measurement at the prediction time);
* runs the RNN over `x_hist` only, giving hidden states h_j;
* scores `e_j = FCNet([h_j, x_last])`, masked by `rnn_mask_att` so only
  actually-measured steps contribute, then normalised;
* pools `context = sum_j a_j h_j` and predicts from
  `concat([x_last, context])`.

It therefore makes ONE prediction, at the last measurement, and its
"attention over all history" is causal by construction -- there is no future
beyond the prediction time.

Our backbone instead pooled a single context over ALL T steps (query =
`outputs[:, -1:, :]`) and broadcast it to every position, while TCSR reads
`logits[:, 0, :]`. A prediction at t=0 thus consumed the entire trajectory.
That is a deviation from DDH, not a property of it.

Fix: apply DDH's own mechanism at each landmark t -- query is the
measurement at t, keys/values are hidden states j < t, masked to measured
steps. `causal=False` restores the old behaviour for comparison. Verified
with the leakage probe: perturbing t>=6 leaves t=0..5 bit-identical.

Expect DDH numbers to DROP relative to the leaky version, especially where
predictions are read from t=0 with little history -- the earlier Scania
figures (CI 0.89-0.95) were inflated by future information.
