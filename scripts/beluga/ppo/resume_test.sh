#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --mem=30G
#SBATCH --time=0-00:20
#SBATCH --job-name=ppo_resume_ant_normal_test
#SBATCH --output=%x-%j.out
#SBATCH --mail-user=taghianj@ualberta.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL

parallel < resume_test.txt
