#!/bin/bash

# Create log directory
mkdir -p logs

# One log file for the entire experiment
LOG_FILE="logs/experiment_$(date +%Y%m%d_%H%M%S).log"

# Redirect ALL stdout and stderr to the log file
exec > "$LOG_FILE" 2>&1

echo "Experiment started at: $(date)"
echo "Log file: $LOG_FILE"
echo "========================================"

# Define poison types
POISON_TYPES=("pattern" "adaptivecifar10" "freq_meg_500")

# Define poison ratios
POISON_RATIOS=(0.003 0.05)

# Define 10 different seeds
SEEDS=(0 1 2 3 4 5 6 7 8 9)

# Loop through each seed
for seed in "${SEEDS[@]}"
do
	    for poison_type in "${POISON_TYPES[@]}"
		        do
				        for poison_ratio in "${POISON_RATIOS[@]}"
						        do
								            echo ""
									                echo "========================================"
											            echo "Running:"
												                echo "seed=$seed"
														            echo "poison_type=$poison_type"
															                echo "poison_ratio=$poison_ratio"
																	            echo "Start time: $(date)"
																		                echo "========================================"

																				            python our_method_with_OT.py \
																						                    -t "$poison_type" \
																								                    -class 10 \
																										                    -pb poison \
																												                    -pr "$poison_ratio" \
																														                    -sample 10 \
																																                    -aug_n 6 \
																																		                    -seed "$seed"

																					                echo "Finished at: $(date)"
																							        done
																								    done
																							    done

																							    echo ""
																							    echo "========================================"
																							    echo "Execution completed for all configurations."
																							    echo "Finished at: $(date)"
																							    echo "========================================"
