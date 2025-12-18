#!/bin/bash

#SBATCH --account=booth-caai
#SBATCH --partition=gpu_h100
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --job-name="x2_s2"
#SBATCH --output=experiments/x2_demo/log/s2_%j.log
#SBATCH --error=experiments/x2_demo/log/s2_%j.err

#---------------------------------------------------------------------------------
# Commands to execute

# load the python module
source /project/caai_amzn_cmpr/efithian/Adaptive-Net/.venv/bin/activate

# navigate to project root
cd /project/caai_amzn_cmpr/efithian/Adaptive-Net

# run the training script
python -m experiments.x2_demo.s2_train_lr

