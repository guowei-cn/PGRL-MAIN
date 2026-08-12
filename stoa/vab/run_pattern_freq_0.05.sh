#!/bin/bash

# Create a logs directory if it doesn't exist
mkdir -p logs
#
## Run 1: badnet trigger with 0.0 cover portion
#echo "Running pattern (cover 0.05)..."
#python Train_cifar10.py \
#  --trigger_type pattern \
#  --trigger_label 2 \
#  --trigger_path ./trigger/cifar10/cifar_1.png \
#  --posioned_portion 0.05 \
#  --cover_portion 0.05 \
#  --model_name ResNet18 \
#  | tee logs/badnet_cover0.05.log
#
## Run 2: badnet trigger with 0.0 cover portion
#echo "Running pattern (cover 0.0)..."
#python Train_cifar10.py \
#  --trigger_type pattern \
#  --trigger_label 2 \
#  --trigger_path ./trigger/cifar10/cifar_1.png \
#  --posioned_portion 0.05 \
#  --cover_portion 0.0 \
#  --model_name ResNet18 \
#  | tee logs/badnet_cover0.0.log
#
## Run 3: adaptive trigger with 0.05 cover portion
#echo "Running AdaptiveCIFAR10 (cover 0.05)..."
#python Train_cifar10.py \
#  --trigger_type adaptivecifar10 \
#  --trigger_label 2 \
#  --trigger_path ./trigger/cifar10/hellokitty.png \
#  --posioned_portion 0.05 \
#  --cover_portion 0.05 \
#  --model_name ResNet18 \
#  | tee logs/adaptive_cover0.05.log
#
## Run 4: adaptive trigger with 0.0 cover portion
#echo "Running AdaptiveCIFAR10 (cover 0.0)..."
#python Train_cifar10.py \
#  --trigger_type adaptivecifar10 \
#  --trigger_label 2 \
#  --trigger_path ./trigger/cifar10/hellokitty.png \
#  --posioned_portion 0.05 \
#  --cover_portion 0.0 \
#  --model_name ResNet18 \
#  | tee logs/adaptive_cover0.0.log


## Run 2: badnet trigger with 0.0 cover portion
#echo "Running pattern (cover 0.0)..."
#python Train_cifar10.py \
#  --trigger_type pattern \
#  --trigger_label 2 \
#  --trigger_path ./trigger/cifar10/cifar_1.png \
#  --posioned_portion 0.003 \
#  --cover_portion 0.0 \
#  --model_name ResNet18 \
#  | tee logs/badnet_cover0.0_poison0.003.log
#
## Run 3: adaptive trigger with 0.05 cover portion
#echo "Running AdaptiveCIFAR10 (cover 0.003)..."
#python Train_cifar10.py \
#  --trigger_type adaptivecifar10 \
#  --trigger_label 2 \
#  --trigger_path ./trigger/cifar10/hellokitty.png \
#  --posioned_portion 0.003 \
#  --cover_portion 0.003 \
#  --model_name ResNet18 \
#  | tee logs/adaptive_cover0.003_poison0.003.log


## Run 2: badnet trigger with 0.0 cover portion
#echo "Running freq (posion 0.003 cover 0.0)..."
#python Train_cifar10.py \
#  --trigger_type freq \
#  --trigger_label 2 \
#  --trigger_path ./trigger/cifar10/cifar_1.png \
#  --posioned_portion 0.003 \
#  --cover_portion 0.0 \
#  --model_name ResNet18 \
#  | tee logs/freq_cover0.0_poison0.003.log

# Run 3: adaptive trigger with 0.05 cover portion
echo "Running freq (posion 0.05 cover 0.0)..."
python Train_cifar10.py \
  --trigger_type freq \
  --trigger_label 2 \
  --trigger_path ./trigger/cifar10/hellokitty.png \
  --posioned_portion 0.05 \
  --cover_portion 0.0 \
  --model_name ResNet18 \
  | tee logs/freq_cover0.0_poison0.05.log

echo "All runs completed."
