for i in {0..48..2}
do
  echo "#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:v100l:4
#SBATCH --ntasks-per-node=20
#SBATCH --mem=187G
#SBATCH --time=7-00:00
#SBATCH --job-name=sac_$i
#SBATCH --output=%x-%j.out
#SBATCH --mail-user=sschoepp@ualberta.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL

parallel < s$i.txt" >> s"$i".sh
done

for i in {0..48..2}
do
  j=$((i+1))
  echo "CUDA_VISIBLE_DEVICES=0 python /home/sschoepp/scratch/openai/controllers/sacv2/sacv2_n_controller.py -a -c --resumable -s 0 -tl 7 -ps -pss $i
CUDA_VISIBLE_DEVICES=0 python /home/sschoepp/scratch/openai/controllers/sacv2/sacv2_n_controller.py -a -c --resumable -s 1 -tl 7 -ps -pss $i
CUDA_VISIBLE_DEVICES=0 python /home/sschoepp/scratch/openai/controllers/sacv2/sacv2_n_controller.py -a -c --resumable -s 2 -tl 7 -ps -pss $i
CUDA_VISIBLE_DEVICES=0 python /home/sschoepp/scratch/openai/controllers/sacv2/sacv2_n_controller.py -a -c --resumable -s 3 -tl 7 -ps -pss $i
CUDA_VISIBLE_DEVICES=0 python /home/sschoepp/scratch/openai/controllers/sacv2/sacv2_n_controller.py -a -c --resumable -s 4 -tl 7 -ps -pss $i
CUDA_VISIBLE_DEVICES=1 python /home/sschoepp/scratch/openai/controllers/sacv2/sacv2_n_controller.py -a -c --resumable -s 5 -tl 7 -ps -pss $i
CUDA_VISIBLE_DEVICES=1 python /home/sschoepp/scratch/openai/controllers/sacv2/sacv2_n_controller.py -a -c --resumable -s 6 -tl 7 -ps -pss $i
CUDA_VISIBLE_DEVICES=1 python /home/sschoepp/scratch/openai/controllers/sacv2/sacv2_n_controller.py -a -c --resumable -s 7 -tl 7 -ps -pss $i
CUDA_VISIBLE_DEVICES=1 python /home/sschoepp/scratch/openai/controllers/sacv2/sacv2_n_controller.py -a -c --resumable -s 8 -tl 7 -ps -pss $i
CUDA_VISIBLE_DEVICES=1 python /home/sschoepp/scratch/openai/controllers/sacv2/sacv2_n_controller.py -a -c --resumable -s 9 -tl 7 -ps -pss $i
CUDA_VISIBLE_DEVICES=2 python /home/sschoepp/scratch/openai/controllers/sacv2/sacv2_n_controller.py -a -c --resumable -s 0 -tl 7 -ps -pss $j
CUDA_VISIBLE_DEVICES=2 python /home/sschoepp/scratch/openai/controllers/sacv2/sacv2_n_controller.py -a -c --resumable -s 1 -tl 7 -ps -pss $j
CUDA_VISIBLE_DEVICES=2 python /home/sschoepp/scratch/openai/controllers/sacv2/sacv2_n_controller.py -a -c --resumable -s 2 -tl 7 -ps -pss $j
CUDA_VISIBLE_DEVICES=2 python /home/sschoepp/scratch/openai/controllers/sacv2/sacv2_n_controller.py -a -c --resumable -s 3 -tl 7 -ps -pss $j
CUDA_VISIBLE_DEVICES=2 python /home/sschoepp/scratch/openai/controllers/sacv2/sacv2_n_controller.py -a -c --resumable -s 4 -tl 7 -ps -pss $j
CUDA_VISIBLE_DEVICES=3 python /home/sschoepp/scratch/openai/controllers/sacv2/sacv2_n_controller.py -a -c --resumable -s 5 -tl 7 -ps -pss $j
CUDA_VISIBLE_DEVICES=3 python /home/sschoepp/scratch/openai/controllers/sacv2/sacv2_n_controller.py -a -c --resumable -s 6 -tl 7 -ps -pss $j
CUDA_VISIBLE_DEVICES=3 python /home/sschoepp/scratch/openai/controllers/sacv2/sacv2_n_controller.py -a -c --resumable -s 7 -tl 7 -ps -pss $j
CUDA_VISIBLE_DEVICES=3 python /home/sschoepp/scratch/openai/controllers/sacv2/sacv2_n_controller.py -a -c --resumable -s 8 -tl 7 -ps -pss $j
CUDA_VISIBLE_DEVICES=3 python /home/sschoepp/scratch/openai/controllers/sacv2/sacv2_n_controller.py -a -c --resumable -s 9 -tl 7 -ps -pss $j" >> s"$i".txt
done

#for i in {50..98..2}
#do
#  echo "#!/bin/bash
#
##SBATCH --nodes=1
##SBATCH --gres=gpu:v100l:4
##SBATCH --ntasks-per-node=20
##SBATCH --mem=187G
##SBATCH --time=10-00:00
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
#  echo "CUDA_VISIBLE_DEVICES=0 python /home/taghianj/scratch/openai/controllers/sacv2/sacv2_n_controller.py -a -c --resumable -s 0 -tl 10 -ps -pss $i
#CUDA_VISIBLE_DEVICES=0 python /home/taghianj/scratch/openai/controllers/sacv2/sacv2_n_controller.py -a -c --resumable -s 1 -tl 10 -ps -pss $i
#CUDA_VISIBLE_DEVICES=0 python /home/taghianj/scratch/openai/controllers/sacv2/sacv2_n_controller.py -a -c --resumable -s 2 -tl 10 -ps -pss $i
#CUDA_VISIBLE_DEVICES=0 python /home/taghianj/scratch/openai/controllers/sacv2/sacv2_n_controller.py -a -c --resumable -s 3 -tl 10 -ps -pss $i
#CUDA_VISIBLE_DEVICES=0 python /home/taghianj/scratch/openai/controllers/sacv2/sacv2_n_controller.py -a -c --resumable -s 4 -tl 10 -ps -pss $i
#CUDA_VISIBLE_DEVICES=1 python /home/taghianj/scratch/openai/controllers/sacv2/sacv2_n_controller.py -a -c --resumable -s 5 -tl 10 -ps -pss $i
#CUDA_VISIBLE_DEVICES=1 python /home/taghianj/scratch/openai/controllers/sacv2/sacv2_n_controller.py -a -c --resumable -s 6 -tl 10 -ps -pss $i
#CUDA_VISIBLE_DEVICES=1 python /home/taghianj/scratch/openai/controllers/sacv2/sacv2_n_controller.py -a -c --resumable -s 7 -tl 10 -ps -pss $i
#CUDA_VISIBLE_DEVICES=1 python /home/taghianj/scratch/openai/controllers/sacv2/sacv2_n_controller.py -a -c --resumable -s 8 -tl 10 -ps -pss $i
#CUDA_VISIBLE_DEVICES=1 python /home/taghianj/scratch/openai/controllers/sacv2/sacv2_n_controller.py -a -c --resumable -s 9 -tl 10 -ps -pss $i
#CUDA_VISIBLE_DEVICES=2 python /home/taghianj/scratch/openai/controllers/sacv2/sacv2_n_controller.py -a -c --resumable -s 0 -tl 10 -ps -pss $j
#CUDA_VISIBLE_DEVICES=2 python /home/taghianj/scratch/openai/controllers/sacv2/sacv2_n_controller.py -a -c --resumable -s 1 -tl 10 -ps -pss $j
#CUDA_VISIBLE_DEVICES=2 python /home/taghianj/scratch/openai/controllers/sacv2/sacv2_n_controller.py -a -c --resumable -s 2 -tl 10 -ps -pss $j
#CUDA_VISIBLE_DEVICES=2 python /home/taghianj/scratch/openai/controllers/sacv2/sacv2_n_controller.py -a -c --resumable -s 3 -tl 10 -ps -pss $j
#CUDA_VISIBLE_DEVICES=2 python /home/taghianj/scratch/openai/controllers/sacv2/sacv2_n_controller.py -a -c --resumable -s 4 -tl 10 -ps -pss $j
#CUDA_VISIBLE_DEVICES=3 python /home/taghianj/scratch/openai/controllers/sacv2/sacv2_n_controller.py -a -c --resumable -s 5 -tl 10 -ps -pss $j
#CUDA_VISIBLE_DEVICES=3 python /home/taghianj/scratch/openai/controllers/sacv2/sacv2_n_controller.py -a -c --resumable -s 6 -tl 10 -ps -pss $j
#CUDA_VISIBLE_DEVICES=3 python /home/taghianj/scratch/openai/controllers/sacv2/sacv2_n_controller.py -a -c --resumable -s 7 -tl 10 -ps -pss $j
#CUDA_VISIBLE_DEVICES=3 python /home/taghianj/scratch/openai/controllers/sacv2/sacv2_n_controller.py -a -c --resumable -s 8 -tl 10 -ps -pss $j
#CUDA_VISIBLE_DEVICES=3 python /home/taghianj/scratch/openai/controllers/sacv2/sacv2_n_controller.py -a -c --resumable -s 9 -tl 10 -ps -pss $j" >> s"$i".txt
#done