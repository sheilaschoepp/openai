#for i in 33
#do
#  echo "#!/bin/bash
#
##SBATCH --nodes=1
##SBATCH --ntasks-per-node=30
##SBATCH --mem=125G
##SBATCH --time=14-00:00
##SBATCH --job-name=ppo_$i
##SBATCH --output=%x-%j.out
##SBATCH --mail-user=sschoepp@ualberta.ca
##SBATCH --mail-type=BEGIN
##SBATCH --mail-type=END
##SBATCH --mail-type=FAIL
##SBATCH --mail-type=REQUEUE
##SBATCH --mail-type=ALL
#
#parallel < s$i.txt" > s"$i".sh
#
#  echo "python /home/sschoepp/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 0 -tl 14 -ps -pss $i
#python /home/sschoepp/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 1 -tl 14 -ps -pss $i
#python /home/sschoepp/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 2 -tl 14 -ps -pss $i
#python /home/sschoepp/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 3 -tl 14 -ps -pss $i
#python /home/sschoepp/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 4 -tl 14 -ps -pss $i
#python /home/sschoepp/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 5 -tl 14 -ps -pss $i
#python /home/sschoepp/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 6 -tl 14 -ps -pss $i
#python /home/sschoepp/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 7 -tl 14 -ps -pss $i
#python /home/sschoepp/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 8 -tl 14 -ps -pss $i
#python /home/sschoepp/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 9 -tl 14 -ps -pss $i
#python /home/sschoepp/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 10 -tl 14 -ps -pss $i
#python /home/sschoepp/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 11 -tl 14 -ps -pss $i
#python /home/sschoepp/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 12 -tl 14 -ps -pss $i
#python /home/sschoepp/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 13 -tl 14 -ps -pss $i
#python /home/sschoepp/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 14 -tl 14 -ps -pss $i
#python /home/sschoepp/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 15 -tl 14 -ps -pss $i
#python /home/sschoepp/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 16 -tl 14 -ps -pss $i
#python /home/sschoepp/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 17 -tl 14 -ps -pss $i
#python /home/sschoepp/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 18 -tl 14 -ps -pss $i
#python /home/sschoepp/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 19 -tl 14 -ps -pss $i
#python /home/sschoepp/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 20 -tl 14 -ps -pss $i
#python /home/sschoepp/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 21 -tl 14 -ps -pss $i
#python /home/sschoepp/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 22 -tl 14 -ps -pss $i
#python /home/sschoepp/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 23 -tl 14 -ps -pss $i
#python /home/sschoepp/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 24 -tl 14 -ps -pss $i
#python /home/sschoepp/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 25 -tl 14 -ps -pss $i
#python /home/sschoepp/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 26 -tl 14 -ps -pss $i
#python /home/sschoepp/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 27 -tl 14 -ps -pss $i
#python /home/sschoepp/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 28 -tl 14 -ps -pss $i
#python /home/sschoepp/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 29 -tl 14 -ps -pss $i" > s"$i".txt
#done

for i in 33
do
  echo "#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=30
#SBATCH --mem=125G
#SBATCH --time=14-00:00
#SBATCH --job-name=ppo_$i
#SBATCH --output=%x-%j.out
#SBATCH --mail-user=sschoepp@ualberta.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL

parallel < s$i.txt" > s"$i".sh

  echo "python /home/sschoepp/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 0 -tl 14 -ps -pss $i
python /home/sschoepp/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 1 -tl 14 -ps -pss $i
python /home/sschoepp/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 2 -tl 14 -ps -pss $i
python /home/sschoepp/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 3 -tl 14 -ps -pss $i
python /home/sschoepp/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 4 -tl 14 -ps -pss $i
python /home/sschoepp/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 5 -tl 14 -ps -pss $i
python /home/sschoepp/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 6 -tl 14 -ps -pss $i
python /home/sschoepp/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 7 -tl 14 -ps -pss $i
python /home/sschoepp/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 8 -tl 14 -ps -pss $i
python /home/sschoepp/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 9 -tl 14 -ps -pss $i
python /home/sschoepp/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 10 -tl 14 -ps -pss $i
python /home/sschoepp/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 11 -tl 14 -ps -pss $i
python /home/sschoepp/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 12 -tl 14 -ps -pss $i
python /home/sschoepp/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 13 -tl 14 -ps -pss $i
python /home/sschoepp/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 14 -tl 14 -ps -pss $i
python /home/sschoepp/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 15 -tl 14 -ps -pss $i
python /home/sschoepp/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 16 -tl 14 -ps -pss $i
python /home/sschoepp/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 17 -tl 14 -ps -pss $i
python /home/sschoepp/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 18 -tl 14 -ps -pss $i
python /home/sschoepp/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 19 -tl 14 -ps -pss $i
python /home/sschoepp/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 20 -tl 14 -ps -pss $i
python /home/sschoepp/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 21 -tl 14 -ps -pss $i
python /home/sschoepp/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 22 -tl 14 -ps -pss $i
python /home/sschoepp/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 23 -tl 14 -ps -pss $i
python /home/sschoepp/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 24 -tl 14 -ps -pss $i
python /home/sschoepp/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 25 -tl 14 -ps -pss $i
python /home/sschoepp/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 26 -tl 14 -ps -pss $i
python /home/sschoepp/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 27 -tl 14 -ps -pss $i
python /home/sschoepp/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 28 -tl 14 -ps -pss $i
python /home/sschoepp/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 29 -tl 14 -ps -pss $i" > s"$i".txt
done