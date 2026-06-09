#!/bin/bash

# MAX_JOBS=${MAX_JOBS:-$(sysctl -n hw.logicalcpu 2>/dev/null || nproc)}
MAX_JOBS=5

throttle() {
    while [ "$(jobs -rp | wc -l | tr -d ' ')" -ge "$MAX_JOBS" ]; do
        sleep 0.5
    done
}

# Extract config fields needed to build output paths
DATASET=$(python -c "import yaml; c=yaml.safe_load(open('$1')); print(c['dataset_name'])")
ARCH=$(python -c "import yaml; c=yaml.safe_load(open('$1')); print(c['arch']['type'])")
DEF_LAMBDA=$(python -c "import yaml; c=yaml.safe_load(open('$1')); print(c['lambda_'])")
DEF_TARGET_LR=$(python -c "import yaml; c=yaml.safe_load(open('$1')); print(c['target_lr'])")

# For DeepLambdaSA
echo "Running DeepLambdaSA"
for seed in {0..10}; do
    for lambda in 0.1 0.5 0.8 0.95; do
        for target_lr in 0.05 0.1 0.25; do
            result="Results/DeepLambdaSA/${DATASET}/${ARCH}/lambda_${lambda}/landmark_True/target_lr_${target_lr}/exp_1/seed_${seed}/results_test.json"
            if [ -f "$result" ]; then echo "Skipping DeepLambdaSA seed=$seed lambda=$lambda target_lr=$target_lr (done)"; continue; fi
            throttle
            python main.py --seed $seed --config $1 --agent DeepLambdaSA --lambda_ $lambda --target_lr $target_lr --landmark 1 --val_size 0.2 --size 0.2 &
        done
    done
done

# For SA
echo "Running SA"
for seed in {0..10}; do
    for landmark in 0 1; do
        landmark_bool=$([ "$landmark" -eq 1 ] && echo "True" || echo "False")
        result="Results/SA/${DATASET}/${ARCH}/lambda_${DEF_LAMBDA}/landmark_${landmark_bool}/target_lr_${DEF_TARGET_LR}/exp_1/seed_${seed}/results_test.json"
        if [ -f "$result" ]; then echo "Skipping SA seed=$seed landmark=$landmark (done)"; continue; fi
        throttle
        python main.py --seed $seed --config $1 --agent SA --landmark $landmark --val_size 0.2 --size 0.2 &
    done
done

wait
echo "All done"

# # Only for looping through seeds, agent is read from input.
# for seed in {0..4}; do
#     python main.py --seed $seed --config $1 --agent $2
# done
