#!/bin/bash

# MAX_JOBS=${MAX_JOBS:-$(sysctl -n hw.logicalcpu 2>/dev/null || nproc)}
MAX_JOBS=5
DATASET=$(python -c "import yaml; print(yaml.safe_load(open('$1'))['dataset_name'])")

throttle() {
    while [ "$(jobs -rp | wc -l | tr -d ' ')" -ge "$MAX_JOBS" ]; do
        sleep 0.5
    done
}

# For DDH
echo "Running DDH (MAX_JOBS=$MAX_JOBS, dataset=$DATASET)"
for seed in {0..10}; do
    result="DDH_results/${DATASET}/seed_${seed}/results.json"
    if [ -f "$result" ]; then echo "Skipping DDH seed=$seed (done)"; continue; fi
    throttle
    python ddh_minimal.py --seed $seed --config $1 &
done

echo "Running DDH-TC"
for seed in {0..10}; do
    for lambda in 0.1 0.5 0.8 0.95; do
        for target_lr in 0.05 0.1 0.25; do
            result="DDH_results_TC/${DATASET}/seed_${seed}/lambda_${lambda}/target_lr_${target_lr}/results.json"
            if [ -f "$result" ]; then echo "Skipping DDH-TC seed=$seed lambda=$lambda target_lr=$target_lr (done)"; continue; fi
            throttle
            python ddh_minimal_tc.py --seed $seed --config $1 --lambda_ $lambda --target_lr $target_lr &
        done
    done
done

wait
echo "All done"
