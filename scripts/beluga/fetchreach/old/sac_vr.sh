#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:2
#SBATCH --ntasks-per-node=10
#SBATCH --mem=90000M
#SBATCH --time=1-00:00
#SBATCH --job-name=sac_vr
#SBATCH --output=%x-%j.out
#SBATCH --mail-user=sschoepp@ualberta.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL

parallel < sac_vr.txt
