echo "#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:v100l:4
#SBATCH --ntasks-per-node=20
#SBATCH --mem=192000M
#SBATCH --time=10-00:00
#SBATCH --job-name=sac_v4a
#SBATCH --output=%x-%j.out
#SBATCH --mail-user=sschoepp@ualberta.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL

parallel < sac_v4a.txt" > sac_v4a.sh

echo "" > sac_v4a.txt

for s in {0..4}
do
  echo "CUDA_VISIBLE_DEVICES=0 python /home/sschoepp/scratch/openai/controllers/sacv2/sacv2_ab_controller.py -c -e AntEnv-v4 -f /home/sschoepp/scratch/openai/data/SACv2_Ant-v2:20000000_g:0.9646_t:0.0877_a:0.2_lr:0.001092_hd:256_rbs:500000_bs:512_mups:1_tui:1_tef:100000_ee:10_tmsf:1000000_a:True_d:cuda_ps:True_pss:61_resumed/seed$s --resumable -t 20000000 -tl 10" >> sac_v4a.txt
done

for s in {5..9}
do
  echo "CUDA_VISIBLE_DEVICES=1 python /home/sschoepp/scratch/openai/controllers/sacv2/sacv2_ab_controller.py -c -e AntEnv-v4 -f /home/sschoepp/scratch/openai/data/SACv2_Ant-v2:20000000_g:0.9646_t:0.0877_a:0.2_lr:0.001092_hd:256_rbs:500000_bs:512_mups:1_tui:1_tef:100000_ee:10_tmsf:1000000_a:True_d:cuda_ps:True_pss:61_resumed/seed$s --resumable -t 20000000 -tl 10" >> sac_v4a.txt
done

for s in {0..4}
do
  echo "CUDA_VISIBLE_DEVICES=2 python /home/sschoepp/scratch/openai/controllers/sacv2/sacv2_ab_controller.py -c -crb -e AntEnv-v4 -f /home/sschoepp/scratch/openai/data/SACv2_Ant-v2:20000000_g:0.9646_t:0.0877_a:0.2_lr:0.001092_hd:256_rbs:500000_bs:512_mups:1_tui:1_tef:100000_ee:10_tmsf:1000000_a:True_d:cuda_ps:True_pss:61_resumed/seed$s --resumable -t 20000000 -tl 10" >> sac_v4a.txt
done

for s in {5..9}
do
  echo "CUDA_VISIBLE_DEVICES=3 python /home/sschoepp/scratch/openai/controllers/sacv2/sacv2_ab_controller.py -c -crb -e AntEnv-v4 -f /home/sschoepp/scratch/openai/data/SACv2_Ant-v2:20000000_g:0.9646_t:0.0877_a:0.2_lr:0.001092_hd:256_rbs:500000_bs:512_mups:1_tui:1_tef:100000_ee:10_tmsf:1000000_a:True_d:cuda_ps:True_pss:61_resumed/seed$s --resumable -t 20000000 -tl 10" >> sac_v4a.txt
done



echo "#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:v100l:4
#SBATCH --ntasks-per-node=20
#SBATCH --mem=192000M
#SBATCH --time=10-00:00
#SBATCH --job-name=sac_v4b
#SBATCH --output=%x-%j.out
#SBATCH --mail-user=sschoepp@ualberta.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL

parallel < sac_v4b.txt" > sac_v4b.sh

echo "" > sac_v4b.txt

for s in {0..4}
do
  echo "CUDA_VISIBLE_DEVICES=0 python /home/sschoepp/scratch/openai/controllers/sacv2/sacv2_ab_controller.py -c -rn -e AntEnv-v4 -f /home/sschoepp/scratch/openai/data/SACv2_Ant-v2:20000000_g:0.9646_t:0.0877_a:0.2_lr:0.001092_hd:256_rbs:500000_bs:512_mups:1_tui:1_tef:100000_ee:10_tmsf:1000000_a:True_d:cuda_ps:True_pss:61_resumed/seed$s --resumable -t 20000000 -tl 10" >> sac_v4b.txt
done

for s in {5..9}
do
  echo "CUDA_VISIBLE_DEVICES=1 python /home/sschoepp/scratch/openai/controllers/sacv2/sacv2_ab_controller.py -c -rn -e AntEnv-v4 -f /home/sschoepp/scratch/openai/data/SACv2_Ant-v2:20000000_g:0.9646_t:0.0877_a:0.2_lr:0.001092_hd:256_rbs:500000_bs:512_mups:1_tui:1_tef:100000_ee:10_tmsf:1000000_a:True_d:cuda_ps:True_pss:61_resumed/seed$s --resumable -t 20000000 -tl 10" >> sac_v4b.txt
done

for s in {0..4}
do
  echo "CUDA_VISIBLE_DEVICES=2 python /home/sschoepp/scratch/openai/controllers/sacv2/sacv2_ab_controller.py -c -crb -rn -e AntEnv-v4 -f /home/sschoepp/scratch/openai/data/SACv2_Ant-v2:20000000_g:0.9646_t:0.0877_a:0.2_lr:0.001092_hd:256_rbs:500000_bs:512_mups:1_tui:1_tef:100000_ee:10_tmsf:1000000_a:True_d:cuda_ps:True_pss:61_resumed/seed$s --resumable -t 20000000 -tl 10" >> sac_v4b.txt
done

for s in {5..9}
do
  echo "CUDA_VISIBLE_DEVICES=3 python /home/sschoepp/scratch/openai/controllers/sacv2/sacv2_ab_controller.py -c -crb -rn -e AntEnv-v4 -f /home/sschoepp/scratch/openai/data/SACv2_Ant-v2:20000000_g:0.9646_t:0.0877_a:0.2_lr:0.001092_hd:256_rbs:500000_bs:512_mups:1_tui:1_tef:100000_ee:10_tmsf:1000000_a:True_d:cuda_ps:True_pss:61_resumed/seed$s --resumable -t 20000000 -tl 10" >> sac_v4b.txt
done