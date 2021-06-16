for i in {0..48..3}
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

  j=$((i+1))
  k=$((i+2))
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
python /home/sschoepp/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 0 -tl 14 -ps -pss $j
python /home/sschoepp/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 1 -tl 14 -ps -pss $j
python /home/sschoepp/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 2 -tl 14 -ps -pss $j
python /home/sschoepp/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 3 -tl 14 -ps -pss $j
python /home/sschoepp/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 4 -tl 14 -ps -pss $j
python /home/sschoepp/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 5 -tl 14 -ps -pss $j
python /home/sschoepp/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 6 -tl 14 -ps -pss $j
python /home/sschoepp/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 7 -tl 14 -ps -pss $j
python /home/sschoepp/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 8 -tl 14 -ps -pss $j
python /home/sschoepp/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 9 -tl 14 -ps -pss $j
python /home/sschoepp/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 0 -tl 14 -ps -pss $k
python /home/sschoepp/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 1 -tl 14 -ps -pss $k
python /home/sschoepp/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 2 -tl 14 -ps -pss $k
python /home/sschoepp/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 3 -tl 14 -ps -pss $k
python /home/sschoepp/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 4 -tl 14 -ps -pss $k
python /home/sschoepp/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 5 -tl 14 -ps -pss $k
python /home/sschoepp/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 6 -tl 14 -ps -pss $k
python /home/sschoepp/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 7 -tl 14 -ps -pss $k
python /home/sschoepp/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 8 -tl 14 -ps -pss $k
python /home/sschoepp/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 9 -tl 14 -ps -pss $k" > s"$i".txt
done

for i in {51..96..3}
do
  echo "#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=30
#SBATCH --mem=125G
#SBATCH --time=14-00:00
#SBATCH --job-name=ppo_$i
#SBATCH --output=%x-%j.out
#SBATCH --mail-user=taghianj@ualberta.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL

parallel < s$i.txt" > s"$i".sh

  j=$((i+1))
  k=$((i+2))
  echo "python /home/taghianj/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 0 -tl 14 -ps -pss $i
python /home/taghianj/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 1 -tl 14 -ps -pss $i
python /home/taghianj/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 2 -tl 14 -ps -pss $i
python /home/taghianj/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 3 -tl 14 -ps -pss $i
python /home/taghianj/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 4 -tl 14 -ps -pss $i
python /home/taghianj/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 5 -tl 14 -ps -pss $i
python /home/taghianj/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 6 -tl 14 -ps -pss $i
python /home/taghianj/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 7 -tl 14 -ps -pss $i
python /home/taghianj/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 8 -tl 14 -ps -pss $i
python /home/taghianj/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 9 -tl 14 -ps -pss $i
python /home/taghianj/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 0 -tl 14 -ps -pss $j
python /home/taghianj/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 1 -tl 14 -ps -pss $j
python /home/taghianj/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 2 -tl 14 -ps -pss $j
python /home/taghianj/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 3 -tl 14 -ps -pss $j
python /home/taghianj/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 4 -tl 14 -ps -pss $j
python /home/taghianj/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 5 -tl 14 -ps -pss $j
python /home/taghianj/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 6 -tl 14 -ps -pss $j
python /home/taghianj/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 7 -tl 14 -ps -pss $j
python /home/taghianj/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 8 -tl 14 -ps -pss $j
python /home/taghianj/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 9 -tl 14 -ps -pss $j
python /home/taghianj/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 0 -tl 14 -ps -pss $k
python /home/taghianj/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 1 -tl 14 -ps -pss $k
python /home/taghianj/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 2 -tl 14 -ps -pss $k
python /home/taghianj/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 3 -tl 14 -ps -pss $k
python /home/taghianj/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 4 -tl 14 -ps -pss $k
python /home/taghianj/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 5 -tl 14 -ps -pss $k
python /home/taghianj/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 6 -tl 14 -ps -pss $k
python /home/taghianj/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 7 -tl 14 -ps -pss $k
python /home/taghianj/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 8 -tl 14 -ps -pss $k
python /home/taghianj/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 9 -tl 14 -ps -pss $k" > s"$i".txt
done

for i in 99
do
  echo "#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --mem=42G
#SBATCH --time=14-00:00
#SBATCH --job-name=ppo_$i
#SBATCH --output=%x-%j.out
#SBATCH --mail-user=taghianj@ualberta.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL

parallel < s$i.txt" > s"$i".sh

  echo "python /home/taghianj/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 0 -tl 14 -ps -pss $i
python /home/taghianj/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 1 -tl 14 -ps -pss $i
python /home/taghianj/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 2 -tl 14 -ps -pss $i
python /home/taghianj/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 3 -tl 14 -ps -pss $i
python /home/taghianj/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 4 -tl 14 -ps -pss $i
python /home/taghianj/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 5 -tl 14 -ps -pss $i
python /home/taghianj/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 6 -tl 14 -ps -pss $i
python /home/taghianj/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 7 -tl 14 -ps -pss $i
python /home/taghianj/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 8 -tl 14 -ps -pss $i
python /home/taghianj/scratch/openai/controllers/ppov2/ppov2_n_controller.py -lrd -s 9 -tl 14 -ps -pss $i" > s"$i".txt
done
