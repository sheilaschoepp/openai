#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:p100:2
#SBATCH --ntasks-per-node=10
#SBATCH --mem=127518M
#SBATCH --time=0-00:15
#SBATCH --job-name=sac_test
#SBATCH --output=%x-%j.out
#SBATCH --mail-user=sschoepp@ualberta.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL

parallel < sac_test.txt
