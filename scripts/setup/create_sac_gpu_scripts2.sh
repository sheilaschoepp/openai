echo "#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:4
#SBATCH --ntasks-per-node=30
#SBATCH --mem=191000M
#SBATCH --time=7-00:00
#SBATCH --job-name=sac_v1a
#SBATCH --output=%x-%j.out
#SBATCH --mail-user=sschoepp@ualberta.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL

parallel < sac_v1a.txt" > sac_v1a.sh

echo "" > sac_v1a.txt

for s in {25..29}
do
  echo "CUDA_VISIBLE_DEVICES=0 python /home/sschoepp/scratch/openai/controllers/sacv2/mod/sacv2_ab_controller.py -c -e AntEnv-v1 -f /home/sschoepp/scratch/openai/data/SACv2_Ant-v2:20000000_g:0.9646_t:0.0877_a:0.2_lr:0.001092_hd:256_rbs:500000_bs:512_mups:1_tui:1_tef:100000_ee:10_tmsf:100000_a:True_d:cuda_ps:True_pss:61_resumed_mod/seed$s -t 2000000 -tl 7 --resumable" >> sac_v1a.txt
done

for s in {25..29}
do
  echo "CUDA_VISIBLE_DEVICES=1 python /home/sschoepp/scratch/openai/controllers/sacv2/mod/sacv2_ab_controller.py -c -e AntEnv-v1 -f /home/sschoepp/scratch/openai/data/SACv2_Ant-v2:20000000_g:0.9646_t:0.0877_a:0.2_lr:0.001092_hd:256_rbs:500000_bs:512_mups:1_tui:1_tef:100000_ee:10_tmsf:100000_a:True_d:cuda_ps:True_pss:61_resumed_mod/seed$s -t 2000000 -tl 7 --resumable -crb" >> sac_v1a.txt
done

for s in {25..29}
do
  echo "CUDA_VISIBLE_DEVICES=2 python /home/sschoepp/scratch/openai/controllers/sacv2/mod/sacv2_ab_controller.py -c -e AntEnv-v1 -f /home/sschoepp/scratch/openai/data/SACv2_Ant-v2:20000000_g:0.9646_t:0.0877_a:0.2_lr:0.001092_hd:256_rbs:500000_bs:512_mups:1_tui:1_tef:100000_ee:10_tmsf:100000_a:True_d:cuda_ps:True_pss:61_resumed_mod/seed$s -t 2000000 -tl 7 --resumable -rn" >> sac_v1a.txt
done

for s in {25..29}
do
  echo "CUDA_VISIBLE_DEVICES=3 python /home/sschoepp/scratch/openai/controllers/sacv2/mod/sacv2_ab_controller.py -c -e AntEnv-v1 -f /home/sschoepp/scratch/openai/data/SACv2_Ant-v2:20000000_g:0.9646_t:0.0877_a:0.2_lr:0.001092_hd:256_rbs:500000_bs:512_mups:1_tui:1_tef:100000_ee:10_tmsf:100000_a:True_d:cuda_ps:True_pss:61_resumed_mod/seed$s -t 2000000 -tl 7 --resumable -crb -rn" >> sac_v1a.txt
done

echo "#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:4
#SBATCH --ntasks-per-node=30
#SBATCH --mem=191000M
#SBATCH --time=7-00:00
#SBATCH --job-name=sac_v2a
#SBATCH --output=%x-%j.out
#SBATCH --mail-user=sschoepp@ualberta.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL

parallel < sac_v2a.txt" > sac_v2a.sh

echo "" > sac_v2a.txt

for s in {25..29}
do
  echo "CUDA_VISIBLE_DEVICES=0 python /home/sschoepp/scratch/openai/controllers/sacv2/mod/sacv2_ab_controller.py -c -e AntEnv-v2 -f /home/sschoepp/scratch/openai/data/SACv2_Ant-v2:20000000_g:0.9646_t:0.0877_a:0.2_lr:0.001092_hd:256_rbs:500000_bs:512_mups:1_tui:1_tef:100000_ee:10_tmsf:100000_a:True_d:cuda_ps:True_pss:61_resumed_mod/seed$s -t 2000000 -tl 7 --resumable" >> sac_v2a.txt
done

for s in {25..29}
do
  echo "CUDA_VISIBLE_DEVICES=1 python /home/sschoepp/scratch/openai/controllers/sacv2/mod/sacv2_ab_controller.py -c -e AntEnv-v2 -f /home/sschoepp/scratch/openai/data/SACv2_Ant-v2:20000000_g:0.9646_t:0.0877_a:0.2_lr:0.001092_hd:256_rbs:500000_bs:512_mups:1_tui:1_tef:100000_ee:10_tmsf:100000_a:True_d:cuda_ps:True_pss:61_resumed_mod/seed$s -t 2000000 -tl 7 --resumable -crb" >> sac_v2a.txt
done

for s in {25..29}
do
  echo "CUDA_VISIBLE_DEVICES=2 python /home/sschoepp/scratch/openai/controllers/sacv2/mod/sacv2_ab_controller.py -c -e AntEnv-v2 -f /home/sschoepp/scratch/openai/data/SACv2_Ant-v2:20000000_g:0.9646_t:0.0877_a:0.2_lr:0.001092_hd:256_rbs:500000_bs:512_mups:1_tui:1_tef:100000_ee:10_tmsf:100000_a:True_d:cuda_ps:True_pss:61_resumed_mod/seed$s -t 2000000 -tl 7 --resumable -rn" >> sac_v2a.txt
done

for s in {25..29}
do
  echo "CUDA_VISIBLE_DEVICES=3 python /home/sschoepp/scratch/openai/controllers/sacv2/mod/sacv2_ab_controller.py -c -e AntEnv-v2 -f /home/sschoepp/scratch/openai/data/SACv2_Ant-v2:20000000_g:0.9646_t:0.0877_a:0.2_lr:0.001092_hd:256_rbs:500000_bs:512_mups:1_tui:1_tef:100000_ee:10_tmsf:100000_a:True_d:cuda_ps:True_pss:61_resumed_mod/seed$s -t 2000000 -tl 7 --resumable -crb -rn" >> sac_v2a.txt
done

echo "#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:4
#SBATCH --ntasks-per-node=30
#SBATCH --mem=191000M
#SBATCH --time=7-00:00
#SBATCH --job-name=sac_v3a
#SBATCH --output=%x-%j.out
#SBATCH --mail-user=sschoepp@ualberta.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL

parallel < sac_v3a.txt" > sac_v3a.sh

echo "" > sac_v3a.txt

for s in {25..29}
do
  echo "CUDA_VISIBLE_DEVICES=0 python /home/sschoepp/scratch/openai/controllers/sacv2/mod/sacv2_ab_controller.py -c -e AntEnv-v3 -f /home/sschoepp/scratch/openai/data/SACv2_Ant-v2:20000000_g:0.9646_t:0.0877_a:0.2_lr:0.001092_hd:256_rbs:500000_bs:512_mups:1_tui:1_tef:100000_ee:10_tmsf:100000_a:True_d:cuda_ps:True_pss:61_resumed_mod/seed$s -t 2000000 -tl 7 --resumable" >> sac_v3a.txt
done

for s in {25..29}
do
  echo "CUDA_VISIBLE_DEVICES=1 python /home/sschoepp/scratch/openai/controllers/sacv2/mod/sacv2_ab_controller.py -c -e AntEnv-v3 -f /home/sschoepp/scratch/openai/data/SACv2_Ant-v2:20000000_g:0.9646_t:0.0877_a:0.2_lr:0.001092_hd:256_rbs:500000_bs:512_mups:1_tui:1_tef:100000_ee:10_tmsf:100000_a:True_d:cuda_ps:True_pss:61_resumed_mod/seed$s -t 2000000 -tl 7 --resumable -crb" >> sac_v3a.txt
done

for s in {25..29}
do
  echo "CUDA_VISIBLE_DEVICES=2 python /home/sschoepp/scratch/openai/controllers/sacv2/mod/sacv2_ab_controller.py -c -e AntEnv-v3 -f /home/sschoepp/scratch/openai/data/SACv2_Ant-v2:20000000_g:0.9646_t:0.0877_a:0.2_lr:0.001092_hd:256_rbs:500000_bs:512_mups:1_tui:1_tef:100000_ee:10_tmsf:100000_a:True_d:cuda_ps:True_pss:61_resumed_mod/seed$s -t 2000000 -tl 7 --resumable -rn" >> sac_v3a.txt
done

for s in {25..29}
do
  echo "CUDA_VISIBLE_DEVICES=3 python /home/sschoepp/scratch/openai/controllers/sacv2/mod/sacv2_ab_controller.py -c -e AntEnv-v3 -f /home/sschoepp/scratch/openai/data/SACv2_Ant-v2:20000000_g:0.9646_t:0.0877_a:0.2_lr:0.001092_hd:256_rbs:500000_bs:512_mups:1_tui:1_tef:100000_ee:10_tmsf:100000_a:True_d:cuda_ps:True_pss:61_resumed_mod/seed$s -t 2000000 -tl 7 --resumable -crb -rn" >> sac_v3a.txt
done

echo "#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:4
#SBATCH --ntasks-per-node=30
#SBATCH --mem=191000M
#SBATCH --time=7-00:00
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

for s in {25..29}
do
  echo "CUDA_VISIBLE_DEVICES=0 python /home/sschoepp/scratch/openai/controllers/sacv2/mod/sacv2_ab_controller.py -c -e AntEnv-v4 -f /home/sschoepp/scratch/openai/data/SACv2_Ant-v2:20000000_g:0.9646_t:0.0877_a:0.2_lr:0.001092_hd:256_rbs:500000_bs:512_mups:1_tui:1_tef:100000_ee:10_tmsf:100000_a:True_d:cuda_ps:True_pss:61_resumed_mod/seed$s -t 2000000 -tl 7 --resumable" >> sac_v4a.txt
done

for s in {25..29}
do
  echo "CUDA_VISIBLE_DEVICES=1 python /home/sschoepp/scratch/openai/controllers/sacv2/mod/sacv2_ab_controller.py -c -e AntEnv-v4 -f /home/sschoepp/scratch/openai/data/SACv2_Ant-v2:20000000_g:0.9646_t:0.0877_a:0.2_lr:0.001092_hd:256_rbs:500000_bs:512_mups:1_tui:1_tef:100000_ee:10_tmsf:100000_a:True_d:cuda_ps:True_pss:61_resumed_mod/seed$s -t 2000000 -tl 7 --resumable -crb" >> sac_v4a.txt
done

for s in {25..29}
do
  echo "CUDA_VISIBLE_DEVICES=2 python /home/sschoepp/scratch/openai/controllers/sacv2/mod/sacv2_ab_controller.py -c -e AntEnv-v4 -f /home/sschoepp/scratch/openai/data/SACv2_Ant-v2:20000000_g:0.9646_t:0.0877_a:0.2_lr:0.001092_hd:256_rbs:500000_bs:512_mups:1_tui:1_tef:100000_ee:10_tmsf:100000_a:True_d:cuda_ps:True_pss:61_resumed_mod/seed$s -t 2000000 -tl 7 --resumable -rn" >> sac_v4a.txt
done

for s in {25..29}
do
  echo "CUDA_VISIBLE_DEVICES=3 python /home/sschoepp/scratch/openai/controllers/sacv2/mod/sacv2_ab_controller.py -c -e AntEnv-v4 -f /home/sschoepp/scratch/openai/data/SACv2_Ant-v2:20000000_g:0.9646_t:0.0877_a:0.2_lr:0.001092_hd:256_rbs:500000_bs:512_mups:1_tui:1_tef:100000_ee:10_tmsf:100000_a:True_d:cuda_ps:True_pss:61_resumed_mod/seed$s -t 2000000 -tl 7 --resumable -crb -rn" >> sac_v4a.txt
done