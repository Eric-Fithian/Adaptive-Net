#!/bin/bash

#SBATCH --account=booth-caai
#SBATCH --partition=gpu_h100
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --job-name="x3_openml"
#SBATCH --output=experiments/x3_openml/log/%j.log
#SBATCH --error=experiments/x3_openml/log/%j.err

# load the python module
source /project/caai_amzn_cmpr/efithian/Adaptive-Net/.venv/bin/activate

# navigate to project root
cd /project/caai_amzn_cmpr/efithian/Adaptive-Net

# 1. Data Collection (Run on Meta-Train datasets)
python -m experiments.x3_openml.s1_data_collection

# 2. Train Policy
python -m experiments.x3_openml.s2_train_policy

# 3. Meta-Test Evaluation
python -m experiments.x3_openml.s3_evaluation

# 4. Analysis
python -m experiments.x3_openml.s4_analysis

