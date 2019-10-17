#!/bin/bash

# ************************************
# MODIFY THESE OPTIONS
# Modify this: which gpu (according to nvidia-smi) do you want to use for training
# this can be a single number, or a list. E.g "3" or "0,1" "0,2,3"
# the training script will use all gpus you list
GPU="0"

# how large is an epoch, or sub/super epoch test dataset evaluation
test_every_n_step=1000
batch_size=32

# where are your lmdb databases
input_data_folder="/path/to/input/directory/where/lmdb/are/saved"
# where do you want the outputs saved
output_folder="/path/to/output/directory/where/results/are/saved"

# what are the input lmdb databases called
train_lmdb_file="train-mnist.lmdb"
test_lmdb_file="test-mnist.lmdb"

# what learning rate should the network use
learning_rate=3e-4 # Karpathy Constant

use_augmentation=1 # {0, 1}
# MODIFY THESE OPTIONS
# ************************************

# DO NOT MODIFY ANYTHING BELOW

# limit the script to only the GPUs you selected above
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES=${GPU}

mkdir -p ${output_folder}

# launch training script with required options
echo "Launching Training Script"
python3 train_resnet50.py --test_every_n_steps=${test_every_n_step} --batch_size=${batch_size} --train_database="$input_data_folder/$train_lmdb_file" --test_database="$input_data_folder/$test_lmdb_file" --output_folder=${output_folder} --use_augmentation=${use_augmentation} | tee "$output_folder/log.txt"

echo "Job completed"
