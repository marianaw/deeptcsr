#!/bin/zsh
# Small-data learning-curve benchmark: metrics vs number of training
# sequences n in {10,20,30,50,75,100}, fixed test set (test_seed=1234),
# 10 seeds resampling train/val. lambda_=0 throughout.
# Algorithms: baseline (landmark MLE), fitted TCSR, Inc-TCSR (tau=1),
# D-TCSR (tau < 1, selected on validation C-index per seed/size).
set -e
cd "$(dirname "$0")/.."

SEEDS="range(0,10)"
DATASETS="aids,pbc2,rw"
TAUS="0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.95"
TEST_SEED=1234

run_size() {
  local n=$1
  local out=outputs_small/ntrain_$n
  local common=(test_seed=$TEST_SEED n_train=$n output_dir=$out "seed=$SEEDS")

  uv run python scripts/run.py -m dataset=$DATASETS algorithm=cox backbone=linear \
      algorithm.loss_norm=mean $common
  uv run python scripts/run.py -m dataset=$DATASETS algorithm=d_tcsr \
      algorithm.name=inc_tcsr algorithm.target_lr=1.0 $common
  uv run python scripts/run.py -m dataset=$DATASETS algorithm=d_tcsr \
      algorithm.target_lr=$TAUS $common
  for ds in aids pbc2 rw; do
    for seed in $(seq 0 9); do
      uv run python scripts/run_fitted_tcsr.py --dataset $ds --seed $seed \
          --test-seed $TEST_SEED --n-train $n --output-dir $out
    done
  done
}

for n in 10 20 30 50 75 100; do
  run_size $n > /tmp/small_data_n$n.log 2>&1 &
done
wait
echo "all sizes done"
