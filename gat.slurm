#!/bin/bash

#SBATCH --job-name=hyperopt
#SBATCH --ntasks=8
#SBATCH --gres=gpu:0  

module load anaconda
module load cuda/11.7


source /opt/anaconda/etc/profile.d/conda.sh

conda activate /opt/anaconda/envs/torch_gcnn

jupyter nbconvert --to notebook --execute --inplace hyperopt.ipynb

conda deactivate