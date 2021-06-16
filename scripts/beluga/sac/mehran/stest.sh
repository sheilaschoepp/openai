#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:4
#SBATCH --ntasks-per-node=20
#SBATCH --mem=191000M
#SBATCH --time=0-0:10
#SBATCH --job-name=sac_test
#SBATCH --output=%x-%j.out
#SBATCH --mail-user=taghianj@ualberta.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL

parallel < stest.txt
