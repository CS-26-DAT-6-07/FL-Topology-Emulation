#!/bin/bash

#SBATCH --job-name=Xception_FL      # Name of the job
#SBATCH --output=fl_output.log      # Standard output log
#SBATCH --error=fl_error.log        # Error log
#SBATCH --mem=24G                   # 24G is recommended for large models
#SBATCH --cpus-per-task=8           # Extra CPUs to speed up image augmentation
#SBATCH --gres=gpu:1                # Request 1 GPU
#SBATCH --time=04:00:00             # 4 hours (gives enough time for 3 rounds)

# 1. Activate your virtual environment
source .venv/bin/activate

# 2. Run the Flower simulation
# The "." tells Flower to look for the pyproject.toml in the current folder
flwr run .