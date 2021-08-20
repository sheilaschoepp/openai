#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:4
#SBATCH --ntasks-per-node=30
#SBATCH --mem=191000M
#SBATCH --time=7-00:00
#SBATCH --job-name=sac_v1d
#SBATCH --output=%x-%j.out
#SBATCH --mail-user=taghianj@ualberta.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL

parallel < sac_v1d.txt
