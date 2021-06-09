#!/bin/bash -l 
##SBATCH --account=dipiazza 
#SBATCH --partition=gpu --qos=gpu_free --gres=gpu:1
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=1 
#SBATCH --mem=60G 
#SBATCH --time=10:00:00 
echo STARTING AT $() 
time srun python Gan.py 
echo FINISHED AT $() 