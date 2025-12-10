#!/bin/bash

#SBATCH --account=booth-caai
#SBATCH --partition=standard
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --job-name="fashionmnist_correlation_analysis"
#SBATCH --output=output_local/x1_s2_%j.log
#SBATCH --error=output_local/x1_s2_%j.err

#---------------------------------------------------------------------------------
# Commands to execute

# load the python module
source /project/caai_amzn_cmpr/efithian/Adaptive-Net/.venv/bin/activate

# navigate to project root
cd /project/caai_amzn_cmpr/efithian/Adaptive-Net

# run the training script
python -m experiments.x1_correlation.s2_analysis

