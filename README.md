# survan

Survival-analysis baselines (Cox, TC-Cox, DDH, TC-DDH) unified under a single
`DeepTCSR` class. Configuration is via Hydra, package management via `uv`.

## Setup

```bash
uv sync
```

Point `data_root` at a directory holding the dataset files referenced by
`configs/dataset/*.yaml` (e.g. `data/NASA.h5`). A symlink works:

```bash
ln -s /path/to/SurvanData data
```

## Running a single experiment

```bash
uv run python scripts/run.py dataset=nasa backbone=gru_attn algorithm=ddh seed=0
```

Config groups (`configs/{dataset,backbone,algorithm}/*.yaml`) select which
dataset, backbone and algorithm/loss recipe to use. Overrides follow Hydra
syntax: `algorithm.lambda_=0.9 algorithm.target_lr=0.1 dataset.num_epochs=50`.

Results land in `outputs/<algorithm>/<dataset>/<backbone>/lambda_<λ>/target_lr_<τ>/landmark_<L>/seed_<s>/`
as `results_val.json` and `results_test.json`.

## Sweeping seeds (and hyper-parameters)

Hydra `--multirun` runs combinations sequentially:

```bash
uv run python scripts/run.py --multirun \
    dataset=nasa backbone=gru_attn algorithm=tc_ddh \
    algorithm.lambda_=0.1,0.5,0.8,0.95 algorithm.target_lr=0.05,0.1,0.25 \
    seed=0,1,2,3,4,5,6,7,8,9,10
```

## Aggregating + plotting

```bash
uv run python scripts/aggregate.py --root outputs --out outputs/results.csv
uv run python scripts/make_plots.py --root outputs --out .
```

`make_plots.py` picks the best-validation hyper-parameter combo per seed and
renders `summary_COX.pdf` (Cox vs TC-Cox) and `summary_DDH.pdf` (DDH vs
TC-DDH) with paired-t significance stars.

## Small-dataset benchmark (fitted TCSR vs Inc-TCSR vs D-TCSR)

The small-data experiments (AIDS, PBC2, small random walk) compare, at
`lambda_=0` (pure one-step bootstrap targets):

- **Baseline**: landmarking MLE (`algorithm=cox backbone=linear algorithm.loss_norm=mean`)
- **Fitted TCSR** (Maystre & Russo, reference implementation): `scripts/run_fitted_tcsr.py`
  (imports the sibling `tdsurv` repo; set `TDSURV_LIB` if it lives elsewhere)
- **Inc-TCSR**: `algorithm=d_tcsr algorithm.name=inc_tcsr algorithm.target_lr=1.0` —
  identical to D-TCSR except the target network is synchronized every step (τ=1)
- **D-TCSR**: `algorithm=d_tcsr algorithm.target_lr=<τ<1>` — τ selected on validation C-index per seed

Build the datasets with `uv run --with pyreadr python scripts/prepare_small_data.py`
(needs `data/small_datasets/raw/{pbc.rda,aids.rda}`, see the script docstring),
run everything with `scripts/run_small_data.sh`, and summarize with
`uv run python scripts/small_data_table.py`.

Note `algorithm.tc=true` enables temporal-consistency bootstrapping explicitly;
without it, `lambda_=0` means "no TC" (the large-data baselines rely on this).

## Algorithm matrix

| Algorithm | `lambda_` | `target_lr` | Ranking | Cov-pred | Loss norm |
|-----------|-----------|-------------|---------|----------|-----------|
| Cox       | 0         | 1.0         | 0       | 0        | mean      |
| TC-Cox    | >0        | <1.0        | 0       | 0        | mean      |
| DDH       | 0         | 1.0         | 1.0     | 0.1      | weighted  |
| TC-DDH    | >0        | <1.0        | 1.0     | 0.1      | weighted  |

All four share the same `DeepTCSR` training loop (`src/survan/model.py`).

## Layout

```
src/survan/        package (model, backbones, losses, data, metrics)
configs/           Hydra config groups + entrypoint config
scripts/           run.py, aggregate.py, make_plots.py
outputs/           per-run results (gitignored)
_legacy_results/   archived legacy results for parity checks (gitignored)
```
