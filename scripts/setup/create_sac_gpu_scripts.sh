echo "#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:4
#SBATCH --ntasks-per-node=30
#SBATCH --mem=191000M
#SBATCH --time=1-00:00
#SBATCH --job-name=sac_a
#SBATCH --output=%x-%j.out
#SBATCH --mail-user=sschoepp@ualberta.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL

parallel < sac_a.txt" > sac_a.sh

echo "" > sac_a.txt

for s in {0..9}
do
  echo "CUDA_VISIBLE_DEVICES=0 python /home/sschoepp/scratch/openai/controllers/sacv2/mod/sacv2_ab_controller.py -c -e FetchReachEnv-v1 -f /home/sschoepp/scratch/openai/data/SACv2_FetchReachEnv-v0:2000000_g:0.8097_t:0.0721_a:0.2_lr:0.001738_hd:256_rbs:10000_bs:512_mups:1_tui:1_tef:10000_ee:10_tmsf:20000_a:True_d:cuda_ps:True_pss:21_mod/seed$s -t 2000000" >> sac_a.txt
done

for s in {0..9}
do
  echo "CUDA_VISIBLE_DEVICES=2 python /home/sschoepp/scratch/openai/controllers/sacv2/mod/sacv2_ab_controller.py -c -e FetchReachEnv-v1 -f /home/sschoepp/scratch/openai/data/SACv2_FetchReachEnv-v0:2000000_g:0.8097_t:0.0721_a:0.2_lr:0.001738_hd:256_rbs:10000_bs:512_mups:1_tui:1_tef:10000_ee:10_tmsf:20000_a:True_d:cuda_ps:True_pss:21_mod/seed$s -t 2000000 -crb" >> sac_a.txt
done

for s in {0..9}
do
  echo "CUDA_VISIBLE_DEVICES=2 python /home/sschoepp/scratch/openai/controllers/sacv2/mod/sacv2_ab_controller.py -c -e FetchReachEnv-v1 -f /home/sschoepp/scratch/openai/data/SACv2_FetchReachEnv-v0:2000000_g:0.8097_t:0.0721_a:0.2_lr:0.001738_hd:256_rbs:10000_bs:512_mups:1_tui:1_tef:10000_ee:10_tmsf:20000_a:True_d:cuda_ps:True_pss:21_mod/seed$s -t 2000000 -crb -rn" >> sac_a.txt
done

echo "#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:4
#SBATCH --ntasks-per-node=30
#SBATCH --mem=191000M
#SBATCH --time=1-00:00
#SBATCH --job-name=sac_b
#SBATCH --output=%x-%j.out
#SBATCH --mail-user=sschoepp@ualberta.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL

parallel < sac_b.txt" > sac_b.sh

echo "" > sac_b.txt

for s in {0..9}
do
  echo "CUDA_VISIBLE_DEVICES=0 python /home/sschoepp/scratch/openai/controllers/sacv2/mod/sacv2_ab_controller.py -c -e FetchReachEnv-v4 -f /home/sschoepp/scratch/openai/data/SACv2_FetchReachEnv-v0:2000000_g:0.8097_t:0.0721_a:0.2_lr:0.001738_hd:256_rbs:10000_bs:512_mups:1_tui:1_tef:10000_ee:10_tmsf:20000_a:True_d:cuda_ps:True_pss:21_mod/seed$s -t 2000000" >> sac_b.txt
done

for s in {0..9}
do
  echo "CUDA_VISIBLE_DEVICES=2 python /home/sschoepp/scratch/openai/controllers/sacv2/mod/sacv2_ab_controller.py -c -e FetchReachEnv-v4 -f /home/sschoepp/scratch/openai/data/SACv2_FetchReachEnv-v0:2000000_g:0.8097_t:0.0721_a:0.2_lr:0.001738_hd:256_rbs:10000_bs:512_mups:1_tui:1_tef:10000_ee:10_tmsf:20000_a:True_d:cuda_ps:True_pss:21_mod/seed$s -t 2000000 -crb" >> sac_b.txt
done

for s in {0..9}
do
  echo "CUDA_VISIBLE_DEVICES=2 python /home/sschoepp/scratch/openai/controllers/sacv2/mod/sacv2_ab_controller.py -c -e FetchReachEnv-v4 -f /home/sschoepp/scratch/openai/data/SACv2_FetchReachEnv-v0:2000000_g:0.8097_t:0.0721_a:0.2_lr:0.001738_hd:256_rbs:10000_bs:512_mups:1_tui:1_tef:10000_ee:10_tmsf:20000_a:True_d:cuda_ps:True_pss:21_mod/seed$s -t 2000000 -crb -rn" >> sac_b.txt
done

echo "#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:4
#SBATCH --ntasks-per-node=30
#SBATCH --mem=191000M
#SBATCH --time=1-00:00
#SBATCH --job-name=sac_c
#SBATCH --output=%x-%j.out
#SBATCH --mail-user=sschoepp@ualberta.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL

parallel < sac_c.txt" > sac_c.sh

echo "" > sac_c.txt

for s in {0..9}
do
  echo "CUDA_VISIBLE_DEVICES=0 python /home/sschoepp/scratch/openai/controllers/sacv2/mod/sacv2_ab_controller.py -c -e FetchReachEnv-v6 -f /home/sschoepp/scratch/openai/data/SACv2_FetchReachEnv-v0:2000000_g:0.8097_t:0.0721_a:0.2_lr:0.001738_hd:256_rbs:10000_bs:512_mups:1_tui:1_tef:10000_ee:10_tmsf:20000_a:True_d:cuda_ps:True_pss:21_mod/seed$s -t 2000000" >> sac_c.txt
done

for s in {0..9}
do
  echo "CUDA_VISIBLE_DEVICES=2 python /home/sschoepp/scratch/openai/controllers/sacv2/mod/sacv2_ab_controller.py -c -e FetchReachEnv-v6 -f /home/sschoepp/scratch/openai/data/SACv2_FetchReachEnv-v0:2000000_g:0.8097_t:0.0721_a:0.2_lr:0.001738_hd:256_rbs:10000_bs:512_mups:1_tui:1_tef:10000_ee:10_tmsf:20000_a:True_d:cuda_ps:True_pss:21_mod/seed$s -t 2000000 -crb" >> sac_c.txt
done

for s in {0..9}
do
  echo "CUDA_VISIBLE_DEVICES=2 python /home/sschoepp/scratch/openai/controllers/sacv2/mod/sacv2_ab_controller.py -c -e FetchReachEnv-v6 -f /home/sschoepp/scratch/openai/data/SACv2_FetchReachEnv-v0:2000000_g:0.8097_t:0.0721_a:0.2_lr:0.001738_hd:256_rbs:10000_bs:512_mups:1_tui:1_tef:10000_ee:10_tmsf:20000_a:True_d:cuda_ps:True_pss:21_mod/seed$s -t 2000000 -crb -rn" >> sac_c.txt
done

echo "#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:4
#SBATCH --ntasks-per-node=30
#SBATCH --mem=191000M
#SBATCH --time=1-00:00
#SBATCH --job-name=sac_d
#SBATCH --output=%x-%j.out
#SBATCH --mail-user=sschoepp@ualberta.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL

parallel < sac_d.txt" > sac_d.sh

echo "" > sac_d.txt

for s in {0..9}
do
  echo "CUDA_VISIBLE_DEVICES=0 python /home/sschoepp/scratch/openai/controllers/sacv2/mod/sacv2_ab_controller.py -c -e FetchReachEnv-v1 -f /home/sschoepp/scratch/openai/data/SACv2_FetchReachEnv-v0:2000000_g:0.8097_t:0.0721_a:0.2_lr:0.001738_hd:256_rbs:10000_bs:512_mups:1_tui:1_tef:10000_ee:10_tmsf:20000_a:True_d:cuda_ps:True_pss:21_mod/seed$s -t 2000000 -rn" >> sac_d.txt
done

for s in {0..9}
do
  echo "CUDA_VISIBLE_DEVICES=2 python /home/sschoepp/scratch/openai/controllers/sacv2/mod/sacv2_ab_controller.py -c -e FetchReachEnv-v4 -f /home/sschoepp/scratch/openai/data/SACv2_FetchReachEnv-v0:2000000_g:0.8097_t:0.0721_a:0.2_lr:0.001738_hd:256_rbs:10000_bs:512_mups:1_tui:1_tef:10000_ee:10_tmsf:20000_a:True_d:cuda_ps:True_pss:21_mod/seed$s -t 2000000 -rn" >> sac_d.txt
done

for s in {0..9}
do
  echo "CUDA_VISIBLE_DEVICES=2 python /home/sschoepp/scratch/openai/controllers/sacv2/mod/sacv2_ab_controller.py -c -e FetchReachEnv-v6 -f /home/sschoepp/scratch/openai/data/SACv2_FetchReachEnv-v0:2000000_g:0.8097_t:0.0721_a:0.2_lr:0.001738_hd:256_rbs:10000_bs:512_mups:1_tui:1_tef:10000_ee:10_tmsf:20000_a:True_d:cuda_ps:True_pss:21_mod/seed$s -t 2000000 -rn" >> sac_d.txt
done