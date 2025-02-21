#!/bin/bash

# For DeepLambdaSA 
echo "Running DeepLambdaSA"
for seed in {0..10}; do
    for lambda in 0.0 0.9 0.95; do
        python main.py --seed $seed --config $1 --agent DeepLambdaSA --lambda_ $lambda --landmark 1
    done
done

# For SA
echo "Running SA"
for seed in {0..10}; do
    for landmark in 0 1; do
        python main.py --seed $seed --config $1 --agent SA --landmark $landmark --lambda_ 0.9
    done
done

# # Only for looping through seeds, agent is read from input.
# for seed in {0..4}; do
#     python main.py --seed $seed --config $1 --agent $2
# done
