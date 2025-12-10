#!/bin/bash

#SBATCH --account=booth-caai
#SBATCH --partition=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --job-name="venv"
#SBATCH --output=output_local/venv_%j.log
#SBATCH --error=output_local/venv_%j.err

#---------------------------------------------------------------------------------
# Commands to execute

# load the python module
source /project/caai_amzn_cmpr/efithian/Adaptive-Net/.venv/bin/activate

# navigate to project root
cd /project/caai_amzn_cmpr/efithian/Adaptive-Net

# run the training script
pip install -r requirements.txt

