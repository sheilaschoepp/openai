#!/bin/bash

#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=5G
#SBATCH --time=0-12:00
#SBATCH --job-name=ppo_test
#SBATCH --output=%x-%j.out
#SBATCH --mail-user=sschoepp@ualberta.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL

python /home/sschoepp/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -t 500000 -s 0 -ps -pss 8