#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --mem=191000M
#SBATCH --time=0-00:20
#SBATCH --job-name=ppo_test
#SBATCH --output=%x-%j.out
#SBATCH --mail-user=sschoepp@ualberta.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL

parallel < ppo_test.txt
