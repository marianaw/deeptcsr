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
