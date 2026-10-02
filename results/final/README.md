# Final large-benchmark results (v2)

4,040 runs: the full tuning grid on seeds 0–9 (2,960 runs) plus seeds 10–29
for every selected configuration (1,080 runs). All arms trained and evaluated
with the same code, under the protocol below. Run outputs (weights, configs,
per-run metrics) live in `outputs_v2/` (not in git).

Datasets: NASA, Scania, LastFM, Large-RW. Families: Logistic-Hazard (LH,
transformer 64×2) and DynamicDeepHit (DDH, GRU+attention 128). Arms per family:
baseline (no TC), Inc-TCSR (τ = 1), D-TCSR (τ < 1).

## What changed since v1 (all v1 numbers are superseded)

1. **Censored subjects were trained as surviving to the horizon** (`data.py`):
   their weights/mask now stop at the censoring time, matching TCSR Eq. 1 and
   the reference `tdsurv` implementation. For λ > 0 the TC λ-return no longer
   bootstraps from padded states past censoring (`losses.py`); the DDH
   baseline and the validation loss also use these weights.
   `tests/test_targets_vs_tdsurv.py` checks our targets/weights against
   `tdsurv` (exact match for λ ∈ {0, 0.5, 0.95, 1}). Every dataset was
   affected, including Large-RW (censoring at the horizon still mislabelled
   the last state and the last horizon step).
2. **Brier(t) is IPCW only.** The unweighted variant counted subjects
   censored before the horizon as events.
3. **C(t) ranks by −log S(Δ)** (same order as 1 − S, but no underflow to exact
   ties at long horizons) and **ties count ½**. NaN now only means "no
   comparable pair" (2.1% of cells, mostly NASA with 40 test subjects).
4. **Grid extended** with τ ∈ {1e-3, 3e-3, 1e-2} (staleness end of the
   tradeoff, for the τ-curve study).

## Protocols

* **TCSR protocol** (`all_runs.csv`): read-out at state 0, global C-index and
  IBS.
* **Dynamic-DeepHit protocol** (`all_landmarks.csv`): conditional risk
  F(t_M + Δ | T > t_M) on a landmark × horizon grid, time-dependent C(t)-index
  and IPCW Brier(t) (censoring KM fitted on the at-risk training subjects).
  Only subjects at risk at t_M are scored. Landmarks are dataset-level
  constants (0 plus quartiles of event/censoring times); horizons are quartiles
  of the remaining time. The grid is shared by all runs so cells can be pooled.

## Grid and selection

λ ∈ {0.1, 0.5, 0.95}; τ ∈ {0.001, 0.003, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75,
0.9, 0.95, 0.99} for D-TCSR, τ = 1 for Inc-TCSR. Up to 1,000 epochs with
early stopping on validation loss, identical for every arm.

One configuration per (algorithm, dataset, metric) is chosen on the **mean
validation score over seeds 0–9** (`scripts/selection.py`), never per seed.
That configuration is then run on seeds 10–29 and **every reported number is
its mean over seeds 0–29** (± = standard error, n = 30). Selected configs:
`selected_configs.csv` (54 configs).

## Headline: TCSR protocol, D-TCSR minus Inc-TCSR (paired over 30 seeds)

| | NASA | Scania | LastFM | Large-RW |
|---|---|---|---|---|
| LH C-index | +0.024 | **+0.067** | **+0.011** | **+0.008** |
| DDH C-index | **+0.018** | **+0.037** | −0.007 | **+0.003** |
| LH IBS | **−0.201** | **−0.009** | **−0.019** | **−0.004** |
| DDH IBS | **−0.024** | −0.023 | −0.001 | −0.001 |

Bold: p < 0.05. D-TCSR is never significantly worse than Inc-TCSR here. LH
Inc-TCSR on NASA (IBS 0.303 vs ≈0.10 for baseline and D-TCSR) is the clearest
instance of same-timescale bootstrapping going unstable.

## Headline: DDH protocol (landmark > 0 × horizon cells)

From `win_counts.md` (34 cells per family × metric). D-TCSR vs Inc-TCSR,
better (significant) / worse (significant):

| | better | worse |
|---|---|---|
| LH C(t) | 34 (32) | 0 (0) |
| DDH C(t) | 32 (27) | 2 (0) |
| LH Brier(t) | 28 (22) | 6 (0) |
| DDH Brier(t) | 31 (27) | 3 (3) |

Inc-TCSR never wins a cell significantly against both other arms.

## Caveats

* **DDH on Scania:** D-TCSR beats Inc-TCSR in 8/9 C(t) cells but the
  untreated DDH baseline wins most cells (circles "○" in `summary_DDH`):
  delayed targets repair Inc-TCSR's instability without improving on plain DDH.
* **Large-RW Brier(t):** smallest gains; D-TCSR is significantly worse than
  Inc-TCSR in 3/7 DDH cells.
* **NASA** has 40 test subjects (10–33 at risk at landmarks > 0); some cells
  have fewer than 30 usable seeds (no comparable pairs).
* **LastFM:** 12 seed-extension runs of LH λ=0.5, τ=0.05 were OOM-killed on the
  pod and re-run successfully; no seeds are missing.

## Files

* `all_runs.csv`, `all_landmarks.csv` — every run (both protocols).
* `tables.txt` — state-0 and landmark tables (`scripts/final_tables.py`).
* `selected_configs.csv`, `select_configs.txt` — selection (`scripts/select_configs.py`).
* `win_counts.md` / `.csv` — per-dataset cell wins (`scripts/win_table.py`).
* Figures (not in git; regenerate locally): `summary_*`
  (`scripts/make_final_plots.py`), `tau_curve_*` (`scripts/plot_tau_curve*.py`).

## Regenerating

```bash
OUTPUT_DIR=outputs_v2 DATA_ROOT=data uv run python scripts/recompute_landmarks.py
OUTPUT_DIR=outputs_v2 uv run python scripts/final_tables.py > results/final/tables.txt
uv run python scripts/select_configs.py
uv run python scripts/make_final_plots.py
uv run python scripts/plot_tau_curve.py && uv run python scripts/plot_tau_curve_fixed.py
uv run python scripts/win_table.py
```

Small-data benchmark (PBC2, AIDS, RW; λ = 0, same τ grid, same selection
rule, 30 seeds): see `results/small_data/`.
