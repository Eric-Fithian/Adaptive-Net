#!/bin/bash

#SBATCH --account=booth-caai
#SBATCH --partition=standard
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --job-name="x3_eda"
#SBATCH --output=experiments/x3_openml/log/%j.log
#SBATCH --error=experiments/x3_openml/log/%j.err

# load the python environment
source /project/caai_amzn_cmpr/efithian/Adaptive-Net/.venv/bin/activate

# navigate to project root
cd /project/caai_amzn_cmpr/efithian/Adaptive-Net

# Run data description / EDA
python -m experiments.x3_openml.s0_data_description

