#!/bin/env bash

# start up an interactive session
srun -p debug -n 1 -t 60:00 --gres=gpu:1 --pty bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda create --name tf2 python=3.6 -y
conda activate tf2

conda install tensorflow2-gpu -y
conda install tensorboard
conda install python-lmdb -y
conda install scikit-image -y
