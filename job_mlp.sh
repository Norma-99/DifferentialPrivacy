#!/bin/bash

#SBATCH --chdir=/scratch/nas/4/norma/DifferentialPrivacy
#SBATCH --output=/scratch/nas/4/norma/.log/stdout-%j.out
#SBATCH --error=/scratch/nas/4/norma/.log/stderr-%j.out
PYTHON="/scratch/nas/4/norma/venv/bin/python"

$PYTHON neural_networks/MLP.py