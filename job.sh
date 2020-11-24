#!/bin/bash

#SBATCH --chdir=/scratch/nas/4/norma/DifferentialPrivacy
#SBATCH --output=/scratch/nas/4/norma/.log/stdout-%j.out
#SBATCH --error=/scratch/nas/4/norma/.log/stderr-%j.out

PYTHON="/scratch/nas/4/norma/venv/bin/python"
CONFIG_FOLDER="/scratch/nas/4/norma/DifferentialPrivacy/configs" 

for i in 1
do
	$PYTHON -m differential_privacy --config=$CONFIG_FOLDER/cent/conf_dffnn.json
done
