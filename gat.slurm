#!/bin/bash

#SBATCH --job-name=edge2
#SBATCH --output=edge2.out
#SBATCH --nodes=1                 
#SBATCH --ntasks-per-node=1       
#SBATCH --cpus-per-task=16   

export CUDA_VISIBLE_DEVICES=0


/home/sergei/.conda/envs/chem/bin/python /home/sergei/Documents/Project/SmartChemDesign/gat/bond.py

