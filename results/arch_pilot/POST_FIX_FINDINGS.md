# Post-fix state of the large benchmarks (2026-09-17)

Two bugs were fixed (details in residual_ab.md):
1. transformer: `MultiHeadAttention(a, a, mask)` passed the causal mask as the
   **value** tensor -> never causal, attention output built from a constant
   triangular matrix. Affected every Cox/transformer run ever produced here,
   legacy included.
2. gru_attn: a single context pooled over ALL T steps was broadcast to every
   position, so a prediction at t=0 consumed the whole trajectory. The real
   Dynamic-DeepHit is causal (predicts once, at the last measurement, over
   history before it). Now applies DDH's mechanism per landmark.

## What the architecture pilot shows after the fixes

| family | config | val CI | val IBS |
|---|---|---|---|
| transformer | h64/l2  | 0.5431 | 0.2976 |
| transformer | h128/l4 | 0.5077 | 0.3092 |
| transformer | h128/l2 | 0.4491 | 0.4395 |
| gru_attn    | h128    | 0.5311 | 0.6379 |
| gru_attn    | h64     | 0.5043 | 0.5272 |

DDH fell from val CI 0.77 (leaky) to ~0.52. Everything on NASA and Scania is
at chance.

## Diagnostics run to distinguish "no signal" from "broken code"

* Same trained model evaluated at landmarks t = 0, 1, 5, 20, 50 on NASA:
  CI 0.62 / 0.53 / 0.49 / 0.54 / 0.54 -- at chance everywhere, so the t=0
  evaluation is not the binding constraint.
* big_rw, where the state is informative by construction (churn depends on
  x.theta): tc_cox CI 0.754, tc_ddh CI 0.731 at only 300 epochs. **The fixed
  code still learns when signal exists.**
* NASA features are already standardised in the h5 (per-feature std ~1,
  max/min std ratio 1.8), so `normalize=False` is correct -- not a
  preprocessing failure.

## Conclusion

The strong NASA/Scania numbers in the current draft were substantially
produced by future-information leakage, not by the models. Predicting from an
initial state that carries little information (a turbofan's first cycle, a
truck's first readout with near-zero accumulated wear) is genuinely close to
impossible.

**The small-data benchmark is unaffected**: it uses the `linear` backbone
(a plain dot product, no attention) and the tdsurv CoxPH reference, neither of
which contains either bug.

## Open decision

* (A) Keep initial-state evaluation and report honestly that only big_rw
  (and possibly LastFM) carries signal.
* (B) Move to landmark evaluation at a later state, which is what the real
  DDH does (predict at the last observed measurement using prior history).
  More informative for longitudinal data; needs target/mask and evaluation
  changes.
* (C) Report both.
