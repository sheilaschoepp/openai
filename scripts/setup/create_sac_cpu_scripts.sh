for i in {2..23..3}
do
  echo "#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=30
#SBATCH --mem=187G
#SBATCH --time=12-00:00
#SBATCH --job-name=sac_$i
#SBATCH --output=%x-%j.out
#SBATCH --mail-user=sschoepp@ualberta.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL

parallel < s$i.txt" >> s"$i".sh

  j=$((i+1))
  k=$((i+2))
  echo "python /home/sschoepp/scratch/openai/controllers/sacv2/sacv2_n_controller.py -a --resumable -s 0 -tl 12 -ps -pss $i
python /home/sschoepp/scratch/openai/controllers/sacv2/sacv2_n_controller.py -a --resumable -s 1 -tl 12 -ps -pss $i
python /home/sschoepp/scratch/openai/controllers/sacv2/sacv2_n_controller.py -a --resumable -s 2 -tl 12 -ps -pss $i
python /home/sschoepp/scratch/openai/controllers/sacv2/sacv2_n_controller.py -a --resumable -s 3 -tl 12 -ps -pss $i
python /home/sschoepp/scratch/openai/controllers/sacv2/sacv2_n_controller.py -a --resumable -s 4 -tl 12 -ps -pss $i
python /home/sschoepp/scratch/openai/controllers/sacv2/sacv2_n_controller.py -a --resumable -s 5 -tl 12 -ps -pss $i
python /home/sschoepp/scratch/openai/controllers/sacv2/sacv2_n_controller.py -a --resumable -s 6 -tl 12 -ps -pss $i
python /home/sschoepp/scratch/openai/controllers/sacv2/sacv2_n_controller.py -a --resumable -s 7 -tl 12 -ps -pss $i
python /home/sschoepp/scratch/openai/controllers/sacv2/sacv2_n_controller.py -a --resumable -s 8 -tl 12 -ps -pss $i
python /home/sschoepp/scratch/openai/controllers/sacv2/sacv2_n_controller.py -a --resumable -s 9 -tl 12 -ps -pss $i
python /home/sschoepp/scratch/openai/controllers/sacv2/sacv2_n_controller.py -a --resumable -s 0 -tl 12 -ps -pss $j
python /home/sschoepp/scratch/openai/controllers/sacv2/sacv2_n_controller.py -a --resumable -s 1 -tl 12 -ps -pss $j
python /home/sschoepp/scratch/openai/controllers/sacv2/sacv2_n_controller.py -a --resumable -s 2 -tl 12 -ps -pss $j
python /home/sschoepp/scratch/openai/controllers/sacv2/sacv2_n_controller.py -a --resumable -s 3 -tl 12 -ps -pss $j
python /home/sschoepp/scratch/openai/controllers/sacv2/sacv2_n_controller.py -a --resumable -s 4 -tl 12 -ps -pss $j
python /home/sschoepp/scratch/openai/controllers/sacv2/sacv2_n_controller.py -a --resumable -s 5 -tl 12 -ps -pss $j
python /home/sschoepp/scratch/openai/controllers/sacv2/sacv2_n_controller.py -a --resumable -s 6 -tl 12 -ps -pss $j
python /home/sschoepp/scratch/openai/controllers/sacv2/sacv2_n_controller.py -a --resumable -s 7 -tl 12 -ps -pss $j
python /home/sschoepp/scratch/openai/controllers/sacv2/sacv2_n_controller.py -a --resumable -s 8 -tl 12 -ps -pss $j
python /home/sschoepp/scratch/openai/controllers/sacv2/sacv2_n_controller.py -a --resumable -s 9 -tl 12 -ps -pss $j
python /home/sschoepp/scratch/openai/controllers/sacv2/sacv2_n_controller.py -a --resumable -s 0 -tl 12 -ps -pss $k
python /home/sschoepp/scratch/openai/controllers/sacv2/sacv2_n_controller.py -a --resumable -s 1 -tl 12 -ps -pss $k
python /home/sschoepp/scratch/openai/controllers/sacv2/sacv2_n_controller.py -a --resumable -s 2 -tl 12 -ps -pss $k
python /home/sschoepp/scratch/openai/controllers/sacv2/sacv2_n_controller.py -a --resumable -s 3 -tl 12 -ps -pss $k
python /home/sschoepp/scratch/openai/controllers/sacv2/sacv2_n_controller.py -a --resumable -s 4 -tl 12 -ps -pss $k
python /home/sschoepp/scratch/openai/controllers/sacv2/sacv2_n_controller.py -a --resumable -s 5 -tl 12 -ps -pss $k
python /home/sschoepp/scratch/openai/controllers/sacv2/sacv2_n_controller.py -a --resumable -s 6 -tl 12 -ps -pss $k
python /home/sschoepp/scratch/openai/controllers/sacv2/sacv2_n_controller.py -a --resumable -s 7 -tl 12 -ps -pss $k
python /home/sschoepp/scratch/openai/controllers/sacv2/sacv2_n_controller.py -a --resumable -s 8 -tl 12 -ps -pss $k
python /home/sschoepp/scratch/openai/controllers/sacv2/sacv2_n_controller.py -a --resumable -s 9 -tl 12 -ps -pss $k" >> s"$i".txt
done

#for i in {50..98..2}
#do
#  echo "#!/bin/bash
#
##SBATCH --nodes=1
##SBATCH --gres=gpu:v100:4
##SBATCH --ntasks-per-node=20
##SBATCH --mem=191000M
##SBATCH --time=7-00:00
##SBATCH --job-name=sac_$i
##SBATCH --output=%x-%j.out
##SBATCH --mail-user=taghianj@ualberta.ca
##SBATCH --mail-type=BEGIN
##SBATCH --mail-type=END
##SBATCH --mail-type=FAIL
##SBATCH --mail-type=REQUEUE
##SBATCH --mail-type=ALL
#
#parallel < s$i.txt" >> s"$i".sh
#done
#
#for i in {50..98..2}
#do
#  j=$((i+1))
#  echo "python /home/taghianj/scratch/openai/controllers/sacv2/sacv2_n_controller.py -a --resumable -s 0 -tl 12 -ps -pss $i
#python /home/taghianj/scratch/openai/controllers/sacv2/sacv2_n_controller.py -a --resumable -s 1 -tl 12 -ps -pss $i
#python /home/taghianj/scratch/openai/controllers/sacv2/sacv2_n_controller.py -a --resumable -s 2 -tl 12 -ps -pss $i
#python /home/taghianj/scratch/openai/controllers/sacv2/sacv2_n_controller.py -a --resumable -s 3 -tl 12 -ps -pss $i
#python /home/taghianj/scratch/openai/controllers/sacv2/sacv2_n_controller.py -a --resumable -s 4 -tl 12 -ps -pss $i
#python /home/taghianj/scratch/openai/controllers/sacv2/sacv2_n_controller.py -a --resumable -s 5 -tl 12 -ps -pss $i
#python /home/taghianj/scratch/openai/controllers/sacv2/sacv2_n_controller.py -a --resumable -s 6 -tl 12 -ps -pss $i
#python /home/taghianj/scratch/openai/controllers/sacv2/sacv2_n_controller.py -a --resumable -s 7 -tl 12 -ps -pss $i
#python /home/taghianj/scratch/openai/controllers/sacv2/sacv2_n_controller.py -a --resumable -s 8 -tl 12 -ps -pss $i
#python /home/taghianj/scratch/openai/controllers/sacv2/sacv2_n_controller.py -a --resumable -s 9 -tl 12 -ps -pss $i
#python /home/taghianj/scratch/openai/controllers/sacv2/sacv2_n_controller.py -a --resumable -s 0 -tl 12 -ps -pss $j
#python /home/taghianj/scratch/openai/controllers/sacv2/sacv2_n_controller.py -a --resumable -s 1 -tl 12 -ps -pss $j
#python /home/taghianj/scratch/openai/controllers/sacv2/sacv2_n_controller.py -a --resumable -s 2 -tl 12 -ps -pss $j
#python /home/taghianj/scratch/openai/controllers/sacv2/sacv2_n_controller.py -a --resumable -s 3 -tl 12 -ps -pss $j
#python /home/taghianj/scratch/openai/controllers/sacv2/sacv2_n_controller.py -a --resumable -s 4 -tl 12 -ps -pss $j
#python /home/taghianj/scratch/openai/controllers/sacv2/sacv2_n_controller.py -a --resumable -s 5 -tl 12 -ps -pss $j
#python /home/taghianj/scratch/openai/controllers/sacv2/sacv2_n_controller.py -a --resumable -s 6 -tl 12 -ps -pss $j
#python /home/taghianj/scratch/openai/controllers/sacv2/sacv2_n_controller.py -a --resumable -s 7 -tl 12 -ps -pss $j
#python /home/taghianj/scratch/openai/controllers/sacv2/sacv2_n_controller.py -a --resumable -s 8 -tl 12 -ps -pss $j
#python /home/taghianj/scratch/openai/controllers/sacv2/sacv2_n_controller.py -a --resumable -s 9 -tl 12 -ps -pss $j" >> s"$i".txt
#done