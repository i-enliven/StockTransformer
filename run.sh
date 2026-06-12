#!/usr/bin/env bash

# Clear or initialize the output file before starting the new experiment
> signal.txt

for run in {1..5}; do 
    echo "=== Starting Run $run/5 ==="
    
    # 1. Run training. If it fails, print an error but do not crash the loop.
    if ! uv run python train.py; then
        echo "Error: Training failed on run $run" >&2
        continue
    fi
    
    # 2. Run inference. Pipefail ensures grep/tee don't mask python crashes.
    set -o pipefail
    if ! uv run python infer.py | grep "Rank" | tee -a signal.txt; then
        echo "Error: Inference or grep failed on run $run" >&2
    fi
    set +o pipefail
    
done

echo -e "\n=== Summary Results ==="
sort -t':' -k3,3 -nr signal.txt | head -n 10

# awk '{count[$3]++} END {for (val in count) print val, count[val]}' signal.txt | sort -k2 -nr | head -n 10
# echo ;
