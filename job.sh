#!/bin/bash

#SBATCH --chdir=/scratch/nas/4/norma/DifferentialPrivacy
#SBATCH --output=/scratch/nas/4/norma/.log/stdout-%j.out
#SBATCH --error=/scratch/nas/4/norma/.log/stderr-%j.out

PYTHON="/scratch/nas/4/norma/venv/bin/python"
CONFIG_FOLDER="/scratch/nas/4/norma/DifferentialPrivacy/configs" 

for i in 3 5 7
do
	$PYTHON -m differential_privacy --config=$CONFIG_FOLDER/hyper/${i}nodes.json
done
