#!/bin/bash

#SBATCH --chdir=/scratch/nas/4/norma/DifferentialPrivacy
#SBATCH --output=/scratch/nas/4/norma/.log/stdout-%j.out
#SBATCH --error=/scratch/nas/4/norma/.log/stderr-%j.out

PYTHON="/scratch/nas/4/norma/venv/bin/python"
CONFIG_FOLDER="/scratch/nas/4/norma/DifferentialPrivacy/configs" 

for i in 1
do
	for j in {1..50}
	do
		$PYTHON -m differential_privacy --config=$CONFIG_FOLDER/adult/${i}nodes.json
		mv ./results/adult/${i}nodes/adult_${i}nodes.csv ./results/adult/${i}nodes/adult_${i}nodes_it${j}.csv
	done
done
