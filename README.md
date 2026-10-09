# D-TCSR

Code to reproduce the experiments of *temporally consistent survival
regression with a delayed target network* (D-TCSR). Baselines (Logistic
Hazard, Dynamic-DeepHit), Inc-TCSR (τ = 1) and D-TCSR (τ < 1) share a single
training loop (`src/survan/model.py`), configured with Hydra.

## Setup

```bash
uv sync          # Python 3.10, CPU JAX; everything runs on CPU
```

The fitted-TCSR baseline of the small-data benchmark uses the reference
implementation of Maystre & Russo. Clone it next to this repository, or point
`TDSURV_LIB` at its `lib/` directory:

```bash
git clone https://github.com/spotify-research/tdsurv ../tdsurv
```

## Data

Every dataset is built into `data/` (the default `data_root`).

| Dataset | Source | Command | Output |
|---|---|---|---|
| PBC2, AIDS, SmallRW | R packages `survival` (`pbc.rda`) and `JM` (`aids.rda`) in `data/small_datasets/raw/`; SmallRW is simulated | `uv run --with pyreadr python scripts/prepare_small_data.py` | `data/small_datasets/*-seqs.pkl` |
| NASA | C-MAPSS turbofan, `train_FD001.txt` and `test_FD001.txt` (NASA Prognostics Data Repository) | `uv run python scripts/prepare_nasa.py --raw <dir>` | `data/NASA.h5` |
| Scania | SCANIA Component X, Swedish National Data Service, DOI [10.5878/bnh5-ka77](https://doi.org/10.5878/bnh5-ka77) (version 3) | `uv run python scripts/prepare_scania.py --raw <dir>` | `data/scania/scania-seqs-h100.pkl` |
| LastFM | [Last.fm 1K](http://ocelma.net/MusicRecommendationDataset/lastfm-1K.html), `userid-timestamp-artid-artname-traid-traname.tsv` | `uv run python scripts/prepare_lastfm.py --raw <tsv>` | `data/lastfm-dataset-1K/` |
| LargeRW | simulated | `uv run python scripts/prepare_large_rw.py` | `data/bigrw-seqs.pkl` |

All scripts are deterministic.

## Single run

```bash
uv run python scripts/run.py dataset=nasa algorithm=tc_cox \
    algorithm.lambda_=0.1 algorithm.target_lr=0.05 seed=0
```

Results (`results_{val,test}.json`, `model.pkl`) are written to
`outputs/<algorithm>/<dataset>/<backbone>/lambda_<λ>/target_lr_<τ>/landmark_<L>/seed_<s>/`.
Finished runs are skipped. The arms of the paper are:

| Arm | LH | DDH |
|---|---|---|
| baseline | `algorithm=cox` | `algorithm=ddh` |
| Inc-TCSR | `algorithm=tc_cox algorithm.name=inc_tc_cox algorithm.target_lr=1.0` | `algorithm=tc_ddh algorithm.name=inc_tc_ddh algorithm.target_lr=1.0` |
| D-TCSR | `algorithm=tc_cox algorithm.target_lr=<τ>` | `algorithm=tc_ddh algorithm.target_lr=<τ>` |

## Reproducing the paper

The drivers run several single-threaded runs in parallel. Set `NWORKERS` to
fit your machine's cores and memory.

**Architecture search** (appendix): 20 runs on NASA and Scania.

```bash
bash scripts/run_arch_pilot.sh                # -> outputs_arch/
uv run python scripts/arch_pilot_table.py     # -> results/arch_pilot/
```

**Large benchmark** (NASA, Scania, LastFM, LargeRW). The tuning grid covers
λ ∈ {0.1, 0.5, 0.95} and 11 values of τ on seeds 0–9 (2,960 runs). One
configuration per (algorithm, dataset, metric) is selected on the mean
validation score, then rerun on seeds 10–29. Every reported number is a mean
over seeds 0–29.

```bash
bash scripts/run_final_sweep.sh                         # seeds 0-9 -> outputs/
uv run python scripts/recompute_landmarks.py            # landmark x horizon grid
uv run python scripts/final_tables.py                   # -> results/final/all_{runs,landmarks}.csv
uv run python scripts/select_configs.py                 # -> results/final/selected_configs.csv
bash scripts/run_seed_extension.sh                      # seeds 10-29
uv run python scripts/recompute_landmarks.py
uv run python scripts/final_tables.py > results/final/tables.txt
uv run python scripts/win_table.py                      # cell-wise wins
uv run python scripts/make_final_plots.py               # summary figures
```

**Small-data benchmark** (PBC2, AIDS, SmallRW; λ = 0). Training-set sizes run
from 10 to 100 sequences.

```bash
bash scripts/run_small_data.sh                          # seeds 0-9 -> outputs_small/
uv run python scripts/small_data_table.py               # selection -> results/small_data/
uv run python scripts/run_small_seed_extension.py -j 4  # seeds 10-29 for the selected configs
uv run python scripts/small_data_table.py
uv run python scripts/plot_small_data.py
```

**τ ablation** (stabilizing effect of the target network): random walks
with H ∈ {30, 50, 100}, 6 values of τ and 30 runs each.

```bash
bash scripts/run_tau_ablation.sh                        # -> results/tau_ablation_main/
```

## Layout

```
src/survan/   model, backbones, losses (temporal-consistency targets), data, metrics
configs/      Hydra config groups: dataset, backbone, algorithm
scripts/      data preparation, run.py, experiment drivers, selection, tables and figures
```
