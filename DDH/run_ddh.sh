#!/bin/bash

# For DDH
echo "Running DDH"
for seed in {0..10}; do
    python ddh_minimal.py --seed $seed --config $1
done