#!/bin/bash

#SBATCH --job-name=spliter
#SBATCH --output=2spliter.out
#SBATCH --nodes=1                 
#SBATCH --ntasks-per-node=1       
#SBATCH --cpus-per-task=16   

export CUDA_VISIBLE_DEVICES=0


#/home/sergei/.conda/envs/chem/bin/python /home/sergei/Documents/SmartChemDesign/gat/prep2.py

/home/sergei/.conda/envs/chem/bin/jupyter nbconvert --to notebook --execute --inplace utils/splitter.ipynb
