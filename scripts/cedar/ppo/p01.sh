#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --mem=125G
#SBATCH --time=10-00:00
#SBATCH --job-name=ppo_8_13
#SBATCH --output=%x-%j.out
#SBATCH --mail-user=sschoepp@ualberta.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL

parallel < p01.txt