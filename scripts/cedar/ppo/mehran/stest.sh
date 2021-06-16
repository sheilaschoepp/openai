#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=30
#SBATCH --mem=125G
#SBATCH --time=00-00:05
#SBATCH --job-name=ppo_test
#SBATCH --output=%x-%j.out
#SBATCH --mail-user=taghianj@ualberta.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL

parallel < stest.txt
