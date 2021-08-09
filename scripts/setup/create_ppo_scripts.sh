echo "#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=30
#SBATCH --mem=191000M
#SBATCH --time=1-00:00
#SBATCH --job-name=ppo_v4a
#SBATCH --output=%x-%j.out
#SBATCH --mail-user=sschoepp@ualberta.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL

parallel < ppo_v4a.txt" > ppo_v4a.sh

for s in {10..29}
do
  echo "python /home/sschoepp/scratch/openai/controllers/ppov2/ppov2_ab_controller.py -e FetchReachEnv-v4 -t 6000000 -f /home/sschoepp/scratch/openai/data/PPOv2_FetchReachEnv-v0:6000000_lr:0.000275_lrd:True_g:0.848_ns:3424_mbs:8_epo:24_eps:0.3_c1:1.0_c2:0.0007_cvl:False_mgn:0.5_gae:True_lam:0.9327_hd:64_lstd:0.0_tef:30000_ee:10_tmsf:60000_d:cpu_ps:True_pss:43/seed$s" >> ppo_v4a.txt
done

for s in {10..19}
do
  echo "python /home/sschoepp/scratch/openai/controllers/ppov2/ppov2_ab_controller.py -e FetchReachEnv-v4 -t 6000000 -f /home/sschoepp/scratch/openai/data/PPOv2_FetchReachEnv-v0:6000000_lr:0.000275_lrd:True_g:0.848_ns:3424_mbs:8_epo:24_eps:0.3_c1:1.0_c2:0.0007_cvl:False_mgn:0.5_gae:True_lam:0.9327_hd:64_lstd:0.0_tef:30000_ee:10_tmsf:60000_d:cpu_ps:True_pss:43/seed$s -cm" >> ppo_v4a.txt
done

echo "#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=30
#SBATCH --mem=191000M
#SBATCH --time=1-00:00
#SBATCH --job-name=ppo_v4b
#SBATCH --output=%x-%j.out
#SBATCH --mail-user=sschoepp@ualberta.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL

parallel < ppo_v4b.txt" > ppo_v4b.sh

for s in {20..29}
do
  echo "python /home/sschoepp/scratch/openai/controllers/ppov2/ppov2_ab_controller.py -e FetchReachEnv-v4 -t 6000000 -f /home/sschoepp/scratch/openai/data/PPOv2_FetchReachEnv-v0:6000000_lr:0.000275_lrd:True_g:0.848_ns:3424_mbs:8_epo:24_eps:0.3_c1:1.0_c2:0.0007_cvl:False_mgn:0.5_gae:True_lam:0.9327_hd:64_lstd:0.0_tef:30000_ee:10_tmsf:60000_d:cpu_ps:True_pss:43/seed$s -cm" >> ppo_v4b.txt
done

for s in {10..29}
do
  echo "python /home/sschoepp/scratch/openai/controllers/ppov2/ppov2_ab_controller.py -e FetchReachEnv-v4 -t 6000000 -f /home/sschoepp/scratch/openai/data/PPOv2_FetchReachEnv-v0:6000000_lr:0.000275_lrd:True_g:0.848_ns:3424_mbs:8_epo:24_eps:0.3_c1:1.0_c2:0.0007_cvl:False_mgn:0.5_gae:True_lam:0.9327_hd:64_lstd:0.0_tef:30000_ee:10_tmsf:60000_d:cpu_ps:True_pss:43/seed$s -rn" >> ppo_v4b.txt
done

echo "#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --mem=191000M
#SBATCH --time=1-00:00
#SBATCH --job-name=ppo_v4c
#SBATCH --output=%x-%j.out
#SBATCH --mail-user=sschoepp@ualberta.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL

parallel < ppo_v4c.txt" > ppo_v4c.sh

for s in {10..29}
do
  echo "python /home/sschoepp/scratch/openai/controllers/ppov2/ppov2_ab_controller.py -e FetchReachEnv-v4 -t 6000000 -f /home/sschoepp/scratch/openai/data/PPOv2_FetchReachEnv-v0:6000000_lr:0.000275_lrd:True_g:0.848_ns:3424_mbs:8_epo:24_eps:0.3_c1:1.0_c2:0.0007_cvl:False_mgn:0.5_gae:True_lam:0.9327_hd:64_lstd:0.0_tef:30000_ee:10_tmsf:60000_d:cpu_ps:True_pss:43/seed$s -cm -rn" >> ppo_v4c.txt
done