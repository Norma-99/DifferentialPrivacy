#!/bin/bash

#SBATCH --chdir=/scratch/nas/4/norma/DifferentialPrivacy
#SBATCH --job-name="DifferentialPrivacy"
#SBATCH --output=/scratch/nas/4/norma/.log/stdout-%j.out
#SBATCH --error=/scratch/nas/4/norma/.log/stderr-%j.out
#SBATCH --wait-all-nodes=1

# Paths
PYTHON="/scratch/nas/4/norma/venv/bin/python"

$PYTHON scripts/generate_dataset_splits.py 14652 3