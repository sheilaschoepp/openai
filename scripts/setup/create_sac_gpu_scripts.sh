echo "#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:v100l:4
#SBATCH --ntasks-per-node=30
#SBATCH --mem=192000M
#SBATCH --time=1-12:00
#SBATCH --job-name=sac_v6a
#SBATCH --output=%x-%j.out
#SBATCH --mail-user=sschoepp@ualberta.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL

parallel < sac_v6a.txt" > sac_v6a.sh

echo "" > sac_v6a.txt

for s in {10..14}
do
  echo "CUDA_VISIBLE_DEVICES=0 python /home/sschoepp/scratch/openai/controllers/sacv2/mod/sacv2_ab_controller.py -c -e FetchReachEnv-v6 -f /home/sschoepp/scratch/openai/data/SACv2_FetchReachEnv-v0:2000000_g:0.8097_t:0.0721_a:0.2_lr:0.001738_hd:256_rbs:10000_bs:512_mups:1_tui:1_tef:10000_ee:10_tmsf:20000_a:True_d:cuda_ps:True_pss:21_mod/seed$s -t 2000000 -tl 1.5" >> sac_v6a.txt
done

for s in {15..16}
do
  echo "CUDA_VISIBLE_DEVICES=0 python /home/sschoepp/scratch/openai/controllers/sacv2/mod/sacv2_ab_controller.py -c -e FetchReachEnv-v6 -f /home/sschoepp/scratch/openai/data/SACv2_FetchReachEnv-v0:2000000_g:0.8097_t:0.0721_a:0.2_lr:0.001738_hd:256_rbs:10000_bs:512_mups:1_tui:1_tef:10000_ee:10_tmsf:20000_a:True_d:cuda_ps:True_pss:21_mod/seed$s -t 2000000 -tl 1.5" >> sac_v6a.txt
done

for s in {17..18}
do
  echo "CUDA_VISIBLE_DEVICES=1 python /home/sschoepp/scratch/openai/controllers/sacv2/mod/sacv2_ab_controller.py -c -e FetchReachEnv-v6 -f /home/sschoepp/scratch/openai/data/SACv2_FetchReachEnv-v0:2000000_g:0.8097_t:0.0721_a:0.2_lr:0.001738_hd:256_rbs:10000_bs:512_mups:1_tui:1_tef:10000_ee:10_tmsf:20000_a:True_d:cuda_ps:True_pss:21_mod/seed$s -t 2000000 -tl 1.5" >> sac_v6a.txt
done

for s in {20..24}
do
  echo "CUDA_VISIBLE_DEVICES=1 python /home/sschoepp/scratch/openai/controllers/sacv2/mod/sacv2_ab_controller.py -c -e FetchReachEnv-v6 -f /home/sschoepp/scratch/openai/data/SACv2_FetchReachEnv-v0:2000000_g:0.8097_t:0.0721_a:0.2_lr:0.001738_hd:256_rbs:10000_bs:512_mups:1_tui:1_tef:10000_ee:10_tmsf:20000_a:True_d:cuda_ps:True_pss:21_mod/seed$s -t 2000000 -tl 1.5" >> sac_v6a.txt
done

for s in {25..29}
do
  echo "CUDA_VISIBLE_DEVICES=2 python /home/sschoepp/scratch/openai/controllers/sacv2/mod/sacv2_ab_controller.py -c -e FetchReachEnv-v6 -f /home/sschoepp/scratch/openai/data/SACv2_FetchReachEnv-v0:2000000_g:0.8097_t:0.0721_a:0.2_lr:0.001738_hd:256_rbs:10000_bs:512_mups:1_tui:1_tef:10000_ee:10_tmsf:20000_a:True_d:cuda_ps:True_pss:21_mod/seed$s -t 2000000 -tl 1.5" >> sac_v6a.txt
done

for s in {10..11}
do
  echo "CUDA_VISIBLE_DEVICES=2 python /home/sschoepp/scratch/openai/controllers/sacv2/mod/sacv2_ab_controller.py -c -crb -e FetchReachEnv-v6 -f /home/sschoepp/scratch/openai/data/SACv2_FetchReachEnv-v0:2000000_g:0.8097_t:0.0721_a:0.2_lr:0.001738_hd:256_rbs:10000_bs:512_mups:1_tui:1_tef:10000_ee:10_tmsf:20000_a:True_d:cuda_ps:True_pss:21_mod/seed$s -t 2000000 -tl 1.5" >> sac_v6a.txt
done

for s in {13..14}
do
  echo "CUDA_VISIBLE_DEVICES=3 python /home/sschoepp/scratch/openai/controllers/sacv2/mod/sacv2_ab_controller.py -c -crb -e FetchReachEnv-v6 -f /home/sschoepp/scratch/openai/data/SACv2_FetchReachEnv-v0:2000000_g:0.8097_t:0.0721_a:0.2_lr:0.001738_hd:256_rbs:10000_bs:512_mups:1_tui:1_tef:10000_ee:10_tmsf:20000_a:True_d:cuda_ps:True_pss:21_mod/seed$s -t 2000000 -tl 1.5" >> sac_v6a.txt
done

for s in {15..19}
do
  echo "CUDA_VISIBLE_DEVICES=3 python /home/sschoepp/scratch/openai/controllers/sacv2/mod/sacv2_ab_controller.py -c -crb -e FetchReachEnv-v6 -f /home/sschoepp/scratch/openai/data/SACv2_FetchReachEnv-v0:2000000_g:0.8097_t:0.0721_a:0.2_lr:0.001738_hd:256_rbs:10000_bs:512_mups:1_tui:1_tef:10000_ee:10_tmsf:20000_a:True_d:cuda_ps:True_pss:21_mod/seed$s -t 2000000 -tl 1.5" >> sac_v6a.txt
done



echo "#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:v100l:4
#SBATCH --ntasks-per-node=30
#SBATCH --mem=192000M
#SBATCH --time=1-12:00
#SBATCH --job-name=sac_v6b
#SBATCH --output=%x-%j.out
#SBATCH --mail-user=sschoepp@ualberta.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL

parallel < sac_v6b.txt" > sac_v6b.sh

echo "" > sac_v6b.txt

for s in {20..24}
do
  echo "CUDA_VISIBLE_DEVICES=0 python /home/sschoepp/scratch/openai/controllers/sacv2/mod/sacv2_ab_controller.py -c -crb -e FetchReachEnv-v6 -f /home/sschoepp/scratch/openai/data/SACv2_FetchReachEnv-v0:2000000_g:0.8097_t:0.0721_a:0.2_lr:0.001738_hd:256_rbs:10000_bs:512_mups:1_tui:1_tef:10000_ee:10_tmsf:20000_a:True_d:cuda_ps:True_pss:21_mod/seed$s -t 2000000 -tl 1.5" >> sac_v6b.txt
done

for s in {25..26}
do
  echo "CUDA_VISIBLE_DEVICES=0 python /home/sschoepp/scratch/openai/controllers/sacv2/mod/sacv2_ab_controller.py -c -crb -e FetchReachEnv-v6 -f /home/sschoepp/scratch/openai/data/SACv2_FetchReachEnv-v0:2000000_g:0.8097_t:0.0721_a:0.2_lr:0.001738_hd:256_rbs:10000_bs:512_mups:1_tui:1_tef:10000_ee:10_tmsf:20000_a:True_d:cuda_ps:True_pss:21_mod/seed$s -t 2000000 -tl 1.5" >> sac_v6b.txt
done

for s in {28..29}
do
  echo "CUDA_VISIBLE_DEVICES=1 python /home/sschoepp/scratch/openai/controllers/sacv2/mod/sacv2_ab_controller.py -c -crb -e FetchReachEnv-v6 -f /home/sschoepp/scratch/openai/data/SACv2_FetchReachEnv-v0:2000000_g:0.8097_t:0.0721_a:0.2_lr:0.001738_hd:256_rbs:10000_bs:512_mups:1_tui:1_tef:10000_ee:10_tmsf:20000_a:True_d:cuda_ps:True_pss:21_mod/seed$s -t 2000000 -tl 1.5" >> sac_v6b.txt
done

for s in {10..14}
do
  echo "CUDA_VISIBLE_DEVICES=1 python /home/sschoepp/scratch/openai/controllers/sacv2/mod/sacv2_ab_controller.py -c -rn -e FetchReachEnv-v6 -f /home/sschoepp/scratch/openai/data/SACv2_FetchReachEnv-v0:2000000_g:0.8097_t:0.0721_a:0.2_lr:0.001738_hd:256_rbs:10000_bs:512_mups:1_tui:1_tef:10000_ee:10_tmsf:20000_a:True_d:cuda_ps:True_pss:21_mod/seed$s -t 2000000 -tl 1.5" >> sac_v6b.txt
done

for s in {15..18}
do
  echo "CUDA_VISIBLE_DEVICES=2 python /home/sschoepp/scratch/openai/controllers/sacv2/mod/sacv2_ab_controller.py -c -rn -e FetchReachEnv-v6 -f /home/sschoepp/scratch/openai/data/SACv2_FetchReachEnv-v0:2000000_g:0.8097_t:0.0721_a:0.2_lr:0.001738_hd:256_rbs:10000_bs:512_mups:1_tui:1_tef:10000_ee:10_tmsf:20000_a:True_d:cuda_ps:True_pss:21_mod/seed$s -t 2000000 -tl 1.5" >> sac_v6b.txt
done

for s in {20..22}
do
  echo "CUDA_VISIBLE_DEVICES=2 python /home/sschoepp/scratch/openai/controllers/sacv2/mod/sacv2_ab_controller.py -c -rn -e FetchReachEnv-v6 -f /home/sschoepp/scratch/openai/data/SACv2_FetchReachEnv-v0:2000000_g:0.8097_t:0.0721_a:0.2_lr:0.001738_hd:256_rbs:10000_bs:512_mups:1_tui:1_tef:10000_ee:10_tmsf:20000_a:True_d:cuda_ps:True_pss:21_mod/seed$s -t 2000000 -tl 1.5" >> sac_v6b.txt
done

for s in {23..24}
do
  echo "CUDA_VISIBLE_DEVICES=3 python /home/sschoepp/scratch/openai/controllers/sacv2/mod/sacv2_ab_controller.py -c -rn -e FetchReachEnv-v6 -f /home/sschoepp/scratch/openai/data/SACv2_FetchReachEnv-v0:2000000_g:0.8097_t:0.0721_a:0.2_lr:0.001738_hd:256_rbs:10000_bs:512_mups:1_tui:1_tef:10000_ee:10_tmsf:20000_a:True_d:cuda_ps:True_pss:21_mod/seed$s -t 2000000 -tl 1.5" >> sac_v6b.txt
done

for s in {25..29}
do
  echo "CUDA_VISIBLE_DEVICES=3 python /home/sschoepp/scratch/openai/controllers/sacv2/mod/sacv2_ab_controller.py -c -rn -e FetchReachEnv-v6 -f /home/sschoepp/scratch/openai/data/SACv2_FetchReachEnv-v0:2000000_g:0.8097_t:0.0721_a:0.2_lr:0.001738_hd:256_rbs:10000_bs:512_mups:1_tui:1_tef:10000_ee:10_tmsf:20000_a:True_d:cuda_ps:True_pss:21_mod/seed$s -t 2000000 -tl 1.5" >> sac_v6b.txt
done

echo "#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:v100l:4
#SBATCH --ntasks-per-node=20
#SBATCH --mem=192000M
#SBATCH --time=1-12:00
#SBATCH --job-name=sac_v6c
#SBATCH --output=%x-%j.out
#SBATCH --mail-user=sschoepp@ualberta.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL

parallel < sac_v6c.txt" > sac_v6c.sh

echo "" > sac_v6c.txt

for s in {10..14}
do
  echo "CUDA_VISIBLE_DEVICES=0 python /home/sschoepp/scratch/openai/controllers/sacv2/mod/sacv2_ab_controller.py -c -crb -rn -e FetchReachEnv-v6 -f /home/sschoepp/scratch/openai/data/SACv2_FetchReachEnv-v0:2000000_g:0.8097_t:0.0721_a:0.2_lr:0.001738_hd:256_rbs:10000_bs:512_mups:1_tui:1_tef:10000_ee:10_tmsf:20000_a:True_d:cuda_ps:True_pss:21_mod/seed$s -t 2000000 -tl 1.5" >> sac_v6c.txt
done

for s in {15..19}
do
  echo "CUDA_VISIBLE_DEVICES=1 python /home/sschoepp/scratch/openai/controllers/sacv2/mod/sacv2_ab_controller.py -c -crb -rn -e FetchReachEnv-v6 -f /home/sschoepp/scratch/openai/data/SACv2_FetchReachEnv-v0:2000000_g:0.8097_t:0.0721_a:0.2_lr:0.001738_hd:256_rbs:10000_bs:512_mups:1_tui:1_tef:10000_ee:10_tmsf:20000_a:True_d:cuda_ps:True_pss:21_mod/seed$s -t 2000000 -tl 1.5" >> sac_v6c.txt
done

for s in {20..24}
do
  echo "CUDA_VISIBLE_DEVICES=2 python /home/sschoepp/scratch/openai/controllers/sacv2/mod/sacv2_ab_controller.py -c -crb -rn -e FetchReachEnv-v6 -f /home/sschoepp/scratch/openai/data/SACv2_FetchReachEnv-v0:2000000_g:0.8097_t:0.0721_a:0.2_lr:0.001738_hd:256_rbs:10000_bs:512_mups:1_tui:1_tef:10000_ee:10_tmsf:20000_a:True_d:cuda_ps:True_pss:21_mod/seed$s -t 2000000 -tl 1.5" >> sac_v6c.txt
done

for s in {25..29}
do
  echo "CUDA_VISIBLE_DEVICES=3 python /home/sschoepp/scratch/openai/controllers/sacv2/mod/sacv2_ab_controller.py -c -crb -rn -e FetchReachEnv-v6 -f /home/sschoepp/scratch/openai/data/SACv2_FetchReachEnv-v0:2000000_g:0.8097_t:0.0721_a:0.2_lr:0.001738_hd:256_rbs:10000_bs:512_mups:1_tui:1_tef:10000_ee:10_tmsf:20000_a:True_d:cuda_ps:True_pss:21_mod/seed$s -t 2000000 -tl 1.5" >> sac_v6c.txt
done

for s in 19
do
  echo "CUDA_VISIBLE_DEVICES=0 python /home/sschoepp/scratch/openai/controllers/sacv2/mod/sacv2_ab_controller.py -c -e FetchReachEnv-v6 -f /home/sschoepp/scratch/openai/data/SACv2_FetchReachEnv-v0:2000000_g:0.8097_t:0.0721_a:0.2_lr:0.001738_hd:256_rbs:10000_bs:512_mups:1_tui:1_tef:10000_ee:10_tmsf:20000_a:True_d:cuda_ps:True_pss:21_mod/seed$s -t 2000000 -tl 1.5" >> sac_v6c.txt
done

for s in 12
do
  echo "CUDA_VISIBLE_DEVICES=1 python /home/sschoepp/scratch/openai/controllers/sacv2/mod/sacv2_ab_controller.py -c -crb -e FetchReachEnv-v6 -f /home/sschoepp/scratch/openai/data/SACv2_FetchReachEnv-v0:2000000_g:0.8097_t:0.0721_a:0.2_lr:0.001738_hd:256_rbs:10000_bs:512_mups:1_tui:1_tef:10000_ee:10_tmsf:20000_a:True_d:cuda_ps:True_pss:21_mod/seed$s -t 2000000 -tl 1.5" >> sac_v6c.txt
done

for s in 27
do
  echo "CUDA_VISIBLE_DEVICES=2 python /home/sschoepp/scratch/openai/controllers/sacv2/mod/sacv2_ab_controller.py -c -crb -e FetchReachEnv-v6 -f /home/sschoepp/scratch/openai/data/SACv2_FetchReachEnv-v0:2000000_g:0.8097_t:0.0721_a:0.2_lr:0.001738_hd:256_rbs:10000_bs:512_mups:1_tui:1_tef:10000_ee:10_tmsf:20000_a:True_d:cuda_ps:True_pss:21_mod/seed$s -t 2000000 -tl 1.5" >> sac_v6c.txt
done

for s in 19
do
  echo "CUDA_VISIBLE_DEVICES=3 python /home/sschoepp/scratch/openai/controllers/sacv2/mod/sacv2_ab_controller.py -c -rn -e FetchReachEnv-v6 -f /home/sschoepp/scratch/openai/data/SACv2_FetchReachEnv-v0:2000000_g:0.8097_t:0.0721_a:0.2_lr:0.001738_hd:256_rbs:10000_bs:512_mups:1_tui:1_tef:10000_ee:10_tmsf:20000_a:True_d:cuda_ps:True_pss:21_mod/seed$s -t 2000000 -tl 1.5" >> sac_v6c.txt
done