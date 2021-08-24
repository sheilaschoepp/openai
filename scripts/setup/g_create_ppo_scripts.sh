echo "#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --mem=100G
#SBATCH --time=7-00:00
#SBATCH --job-name=ppo_v4a1
#SBATCH --output=%x-%j.out
#SBATCH --mail-user=sschoepp@ualberta.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL

parallel < ppo_v4a1.txt" > ppo_v4a1.sh

for s in {0..4}
do
  echo "python /home/sschoepp/scratch/openai/controllers/ppov2/mod2/ppov2_ab_controller.py -e AntEnv-v4 -t 600000000 -f /home/sschoepp/scratch/openai/data/PPOv2_Ant-v2:600000000_lr:0.000123_lrd:True_slrd:0.25_g:0.9839_ns:2471_mbs:1024_epo:5_eps:0.3_c1:1.0_c2:0.0019_cvl:False_mgn:0.5_gae:True_lam:0.911_hd:64_lstd:0.0_tef:3000000_ee:10_tmsf:50000000_d:cpu_ps:True_pss:33_resumed/seed$s --resumable -tl 7" >> ppo_v4a1.txt
done

for s in {5..9}
do
  echo "python /home/sschoepp/scratch/openai/controllers/ppov2/mod2/ppov2_ab_controller.py -e AntEnv-v4 -t 600000000 -f /home/sschoepp/scratch/openai/data/PPOv2_Ant-v2:600000000_lr:0.000123_lrd:True_slrd:0.25_g:0.9839_ns:2471_mbs:1024_epo:5_eps:0.3_c1:1.0_c2:0.0019_cvl:False_mgn:0.5_gae:True_lam:0.911_hd:64_lstd:0.0_tef:3000000_ee:10_tmsf:50000000_d:cpu_ps:True_pss:33_resumed/seed$s --resumable -tl 7" >> ppo_v4a1.txt
done

echo "#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --mem=100G
#SBATCH --time=7-00:00
#SBATCH --job-name=ppo_v4a2
#SBATCH --output=%x-%j.out
#SBATCH --mail-user=sschoepp@ualberta.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL

parallel < ppo_v4a2.txt" > ppo_v4a2.sh

for s in {0..4}
do
  echo "python /home/sschoepp/scratch/openai/controllers/ppov2/mod2/ppov2_ab_controller.py -e AntEnv-v4 -t 600000000 -f /home/sschoepp/scratch/openai/data/PPOv2_Ant-v2:600000000_lr:0.000123_lrd:True_slrd:0.25_g:0.9839_ns:2471_mbs:1024_epo:5_eps:0.3_c1:1.0_c2:0.0019_cvl:False_mgn:0.5_gae:True_lam:0.911_hd:64_lstd:0.0_tef:3000000_ee:10_tmsf:50000000_d:cpu_ps:True_pss:33_resumed/seed$s --resumable -tl 7 -cm" >> ppo_v4a2.txt
done

for s in {5..9}
do
  echo "python /home/sschoepp/scratch/openai/controllers/ppov2/mod2/ppov2_ab_controller.py -e AntEnv-v4 -t 600000000 -f /home/sschoepp/scratch/openai/data/PPOv2_Ant-v2:600000000_lr:0.000123_lrd:True_slrd:0.25_g:0.9839_ns:2471_mbs:1024_epo:5_eps:0.3_c1:1.0_c2:0.0019_cvl:False_mgn:0.5_gae:True_lam:0.911_hd:64_lstd:0.0_tef:3000000_ee:10_tmsf:50000000_d:cpu_ps:True_pss:33_resumed/seed$s --resumable -tl 7 -cm" >> ppo_v4a2.txt
done

echo "#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --mem=100G
#SBATCH --time=7-00:00
#SBATCH --job-name=ppo_v4b1
#SBATCH --output=%x-%j.out
#SBATCH --mail-user=sschoepp@ualberta.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL

parallel < ppo_v4b1.txt" > ppo_v4b1.sh

for s in {0..4}
do
  echo "python /home/sschoepp/scratch/openai/controllers/ppov2/mod2/ppov2_ab_controller.py -e AntEnv-v4 -t 600000000 -f /home/sschoepp/scratch/openai/data/PPOv2_Ant-v2:600000000_lr:0.000123_lrd:True_slrd:0.25_g:0.9839_ns:2471_mbs:1024_epo:5_eps:0.3_c1:1.0_c2:0.0019_cvl:False_mgn:0.5_gae:True_lam:0.911_hd:64_lstd:0.0_tef:3000000_ee:10_tmsf:50000000_d:cpu_ps:True_pss:33_resumed/seed$s --resumable -tl 7 -rn" >> ppo_v4b1.txt
done

for s in {5..9}
do
  echo "python /home/sschoepp/scratch/openai/controllers/ppov2/mod2/ppov2_ab_controller.py -e AntEnv-v4 -t 600000000 -f /home/sschoepp/scratch/openai/data/PPOv2_Ant-v2:600000000_lr:0.000123_lrd:True_slrd:0.25_g:0.9839_ns:2471_mbs:1024_epo:5_eps:0.3_c1:1.0_c2:0.0019_cvl:False_mgn:0.5_gae:True_lam:0.911_hd:64_lstd:0.0_tef:3000000_ee:10_tmsf:50000000_d:cpu_ps:True_pss:33_resumed/seed$s --resumable -tl 7 -rn" >> ppo_v4b1.txt
done

echo "#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --mem=100G
#SBATCH --time=7-00:00
#SBATCH --job-name=ppo_v4b2
#SBATCH --output=%x-%j.out
#SBATCH --mail-user=sschoepp@ualberta.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL

parallel < ppo_v4b2.txt" > ppo_v4b2.sh

for s in {0..4}
do
  echo "python /home/sschoepp/scratch/openai/controllers/ppov2/mod2/ppov2_ab_controller.py -e AntEnv-v4 -t 600000000 -f /home/sschoepp/scratch/openai/data/PPOv2_Ant-v2:600000000_lr:0.000123_lrd:True_slrd:0.25_g:0.9839_ns:2471_mbs:1024_epo:5_eps:0.3_c1:1.0_c2:0.0019_cvl:False_mgn:0.5_gae:True_lam:0.911_hd:64_lstd:0.0_tef:3000000_ee:10_tmsf:50000000_d:cpu_ps:True_pss:33_resumed/seed$s --resumable -tl 7 -cm -rn" >> ppo_v4b2.txt
done

for s in {5..9}
do
  echo "python /home/sschoepp/scratch/openai/controllers/ppov2/mod2/ppov2_ab_controller.py -e AntEnv-v4 -t 600000000 -f /home/sschoepp/scratch/openai/data/PPOv2_Ant-v2:600000000_lr:0.000123_lrd:True_slrd:0.25_g:0.9839_ns:2471_mbs:1024_epo:5_eps:0.3_c1:1.0_c2:0.0019_cvl:False_mgn:0.5_gae:True_lam:0.911_hd:64_lstd:0.0_tef:3000000_ee:10_tmsf:50000000_d:cpu_ps:True_pss:33_resumed/seed$s --resumable -tl 7 -cm -rn" >> ppo_v4b2.txt
done