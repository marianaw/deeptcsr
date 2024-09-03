#!/bin/bash

for seed in {0..4}; do
    python main.py --seed $seed --config $1 --agent $2
done
