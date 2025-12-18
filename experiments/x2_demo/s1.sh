#!/bin/bash

#SBATCH --account=booth-caai
#SBATCH --partition=gpu_h100
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --job-name="x2_s1"
#SBATCH --output=experiments/x2_demo/log/s1_%j.log
#SBATCH --error=experiments/x2_demo/log/s1_%j.err

#---------------------------------------------------------------------------------
# Commands to execute

# load the python module
source /project/caai_amzn_cmpr/efithian/Adaptive-Net/.venv/bin/activate

# navigate to project root
cd /project/caai_amzn_cmpr/efithian/Adaptive-Net

# run the training script
python -m experiments.x2_demo.s1_dataset_creation

