#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=30
#SBATCH --mem=187G
#SBATCH --time=12-00:00
#SBATCH --job-name=sac_89
#SBATCH --output=%x-%j.out
#SBATCH --mail-user=taghianj@ualberta.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL

parallel < s89.txt
