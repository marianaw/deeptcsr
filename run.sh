#!/bin/bash

for seed in {0..4} do;
    python main.py --seed $seed --config configs/config_bigrw.yaml --agent $1
done
