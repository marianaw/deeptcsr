#!/bin/bash
# Small-data learning-curve benchmark (PBC2, AIDS, SmallRW): metrics vs number
# of training sequences n in {10,20,30,50,75,100}, fixed test set
# (test_seed=1234), seeds 0..9 resampling train/val, lambda = 0 throughout.
# Arms: baseline (landmark MLE), fitted TCSR (needs tdsurv, see README),
# Inc-TCSR (tau = 1), D-TCSR (tau grid below). Selection: small_data_table.py.
# Env: DATA_ROOT, OUTPUT_DIR, DATASETS, SEQUENTIAL=1 (one size at a time).
set -e
cd "$(dirname "$0")/.."

SEEDS="range(0,10)"
DATASETS=${DATASETS:-aids,pbc2,rw}
TAUS="0.001,0.003,0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.95,0.99"
TEST_SEED=1234

run_size() {
  local n=$1
  local out=${OUTPUT_DIR:-outputs_small}/ntrain_$n
  local common=(test_seed=$TEST_SEED n_train=$n output_dir=$out "seed=$SEEDS" data_root=${DATA_ROOT:-data})

  uv run python scripts/run.py -m dataset=$DATASETS algorithm=cox backbone=linear \
      algorithm.loss_norm=mean "${common[@]}"
  uv run python scripts/run.py -m dataset=$DATASETS algorithm=d_tcsr \
      algorithm.name=inc_tcsr algorithm.target_lr=1.0 "${common[@]}"
  uv run python scripts/run.py -m dataset=$DATASETS algorithm=d_tcsr \
      algorithm.target_lr=$TAUS "${common[@]}"
  for ds in aids pbc2 rw; do
    for seed in $(seq 0 9); do
      uv run python scripts/run_fitted_tcsr.py --dataset $ds --seed $seed \
          --test-seed $TEST_SEED --n-train $n --output-dir $out
    done
  done
}

for n in 10 20 30 50 75 100; do
  # each size is one hydra multirun process (~4 GB); SEQUENTIAL=1 runs one at a time
  if [[ -n ${SEQUENTIAL:-} ]]; then
    run_size $n > ${OUTPUT_DIR:-outputs_small}_n$n.log 2>&1
  else
    run_size $n > ${OUTPUT_DIR:-outputs_small}_n$n.log 2>&1 &
  fi
done
wait
echo "all sizes done"
