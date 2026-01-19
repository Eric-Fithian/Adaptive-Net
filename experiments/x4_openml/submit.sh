#!/bin/bash

#SBATCH --account=booth-caai
#SBATCH --partition=gpu_h100
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --job-name="x4_deep_nets"
#SBATCH --output=experiments/x4_openml/log/%j.log
#SBATCH --error=experiments/x4_openml/log/%j.err

# Activate environment
source /project/caai_amzn_cmpr/efithian/Adaptive-Net/.venv/bin/activate

# Navigate to project root
cd /project/caai_amzn_cmpr/efithian/Adaptive-Net

# Print GPU info
echo "=== GPU Information ==="
nvidia-smi
echo ""

# 1. Data Collection (Multi-GPU parallelized)
echo "=== Step 1: Data Collection ==="
python -m experiments.x4_openml.s1_data_collection

# 2. Train Policy
echo "=== Step 2: Train Policy ==="
python -m experiments.x4_openml.s2_train_policy

# 3. Meta-Test Evaluation
echo "=== Step 3: Evaluation ==="
python -m experiments.x4_openml.s3_evaluation

# 4. Analysis
echo "=== Step 4: Analysis ==="
python -m experiments.x4_openml.s4_analysis

echo "=== Experiment Complete ==="
