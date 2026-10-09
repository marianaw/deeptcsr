"""Hyper-parameter selection, done once — standard ML practice.

Pick the configuration with the best MEAN validation score across the TUNING
seeds, then report that configuration's mean test score over every seed it
was run with.

Two rules matter:
  * one config per group, not a different winner per seed (per-seed
    selection is optimistic);
  * selection averages over a fixed seed set (0..TUNING_SEEDS-1), so configs
    that later received the seed extension are not compared on more seeds
    than their rivals.
"""
from __future__ import annotations

CFG = ["lambda_", "target_lr"]
TUNING_SEEDS = 10


def pick_global(df, group_keys, val_col, test_col, maximize,
                tuning_seeds=TUNING_SEEDS):
    """Select on seeds < tuning_seeds; return ALL rows of the winning config."""
    d = df.dropna(subset=[val_col, test_col])
    if d.empty:
        return d
    tune = d[d.seed < tuning_seeds] if "seed" in d.columns else d
    if tune.empty:
        tune = d
    m = tune.groupby(list(group_keys) + CFG, as_index=False)[val_col].mean()
    idx = (m.groupby(list(group_keys))[val_col].idxmax() if maximize
           else m.groupby(list(group_keys))[val_col].idxmin())
    best = m.loc[idx, list(group_keys) + CFG]
    return d.merge(best, on=list(group_keys) + CFG, how="inner")
