#!/bin/bash

# Define poison types # "ultrasonic"  "freq_meg_500" "blto")
POISON_TYPES=("pattern" "adaptivecifar10")

# Define poison ratios
POISON_RATIOS=(0.003 0.05)

# Loop through each poison type
for poison_type in "${POISON_TYPES[@]}"
do
    # Loop through each poison ratio
    for poison_ratio in "${POISON_RATIOS[@]}"
    do
        echo "Running with poison_type=$poison_type, poison_ratio=$poison_ratio"
        python our_method_with_OT_emb_as_act.py -t "$poison_type" -class 10 -pb poison -pr "$poison_ratio" -sample 10 -aug_n 6 -pr "$poison_ratio"
    done
done

echo "Execution completed for all configurations."
