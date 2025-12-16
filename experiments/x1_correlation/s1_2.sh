#!/bin/bash

#SBATCH --account=booth-caai
#SBATCH --partition=gpu_h100
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --job-name="fashionmnist_correlation"
#SBATCH --output=experiments/x1_correlation/log/x1_s1_2_%j.log
#SBATCH --error=experiments/x1_correlation/log/x1_s1_2_%j.err

#---------------------------------------------------------------------------------
# Commands to execute

# load the python module
source /project/caai_amzn_cmpr/efithian/Adaptive-Net/.venv/bin/activate

# navigate to project root
cd /project/caai_amzn_cmpr/efithian/Adaptive-Net

# run the training script
python -m experiments.x1_correlation.s1_2_run_training_cifar10

