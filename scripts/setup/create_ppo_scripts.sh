echo "#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=30
#SBATCH --mem=191000M
#SBATCH --time=7-00:00
#SBATCH --job-name=ppo_v1b
#SBATCH --output=%x-%j.out
#SBATCH --mail-user=sschoepp@ualberta.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL

parallel < ppo_v1b.txt" > ppo_v1b.sh

echo "" > ppo_v1b.txt

for s in {17..23}
do
  echo "python /home/sschoepp/scratch/openai/controllers/ppov2/mod/ppov2_ab_controller.py -e AntEnv-v1 -t 600000000 -f /home/sschoepp/scratch/openai/data/PPOv2_Ant-v2:600000000_lr:0.000123_lrd:True_slrd:0.25_g:0.9839_ns:2471_mbs:1024_epo:5_eps:0.3_c1:1.0_c2:0.0019_cvl:False_mgn:0.5_gae:True_lam:0.911_hd:64_lstd:0.0_tef:3000000_ee:10_tmsf:6000000_d:cpu_ps:True_pss:33/seed$s -tl 7 --resumable" >> ppo_v1b.txt
done

for s in {17..23}
do
  echo "python /home/sschoepp/scratch/openai/controllers/ppov2/mod/ppov2_ab_controller.py -e AntEnv-v1 -t 600000000 -f /home/sschoepp/scratch/openai/data/PPOv2_Ant-v2:600000000_lr:0.000123_lrd:True_slrd:0.25_g:0.9839_ns:2471_mbs:1024_epo:5_eps:0.3_c1:1.0_c2:0.0019_cvl:False_mgn:0.5_gae:True_lam:0.911_hd:64_lstd:0.0_tef:3000000_ee:10_tmsf:6000000_d:cpu_ps:True_pss:33/seed$s -tl 7 --resumable -cm" >> ppo_v1b.txt
done

for s in {17..23}
do
  echo "python /home/sschoepp/scratch/openai/controllers/ppov2/mod/ppov2_ab_controller.py -e AntEnv-v1 -t 600000000 -f /home/sschoepp/scratch/openai/data/PPOv2_Ant-v2:600000000_lr:0.000123_lrd:True_slrd:0.25_g:0.9839_ns:2471_mbs:1024_epo:5_eps:0.3_c1:1.0_c2:0.0019_cvl:False_mgn:0.5_gae:True_lam:0.911_hd:64_lstd:0.0_tef:3000000_ee:10_tmsf:6000000_d:cpu_ps:True_pss:33/seed$s -tl 7 --resumable -rn" >> ppo_v1b.txt
done

for s in {17..23}
do
  echo "python /home/sschoepp/scratch/openai/controllers/ppov2/mod/ppov2_ab_controller.py -e AntEnv-v1 -t 600000000 -f /home/sschoepp/scratch/openai/data/PPOv2_Ant-v2:600000000_lr:0.000123_lrd:True_slrd:0.25_g:0.9839_ns:2471_mbs:1024_epo:5_eps:0.3_c1:1.0_c2:0.0019_cvl:False_mgn:0.5_gae:True_lam:0.911_hd:64_lstd:0.0_tef:3000000_ee:10_tmsf:6000000_d:cpu_ps:True_pss:33/seed$s -tl 7 --resumable -cm -rn" >> ppo_v1b.txt
done

echo "#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=30
#SBATCH --mem=191000M
#SBATCH --time=7-00:00
#SBATCH --job-name=ppo_v2b
#SBATCH --output=%x-%j.out
#SBATCH --mail-user=sschoepp@ualberta.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL

parallel < ppo_v2b.txt" > ppo_v2b.sh

echo "" > ppo_v2b.txt

for s in {17..23}
do
  echo "python /home/sschoepp/scratch/openai/controllers/ppov2/mod/ppov2_ab_controller.py -e AntEnv-v2 -t 600000000 -f /home/sschoepp/scratch/openai/data/PPOv2_Ant-v2:600000000_lr:0.000123_lrd:True_slrd:0.25_g:0.9839_ns:2471_mbs:1024_epo:5_eps:0.3_c1:1.0_c2:0.0019_cvl:False_mgn:0.5_gae:True_lam:0.911_hd:64_lstd:0.0_tef:3000000_ee:10_tmsf:6000000_d:cpu_ps:True_pss:33/seed$s -tl 7 --resumable" >> ppo_v2b.txt
done

for s in {17..23}
do
  echo "python /home/sschoepp/scratch/openai/controllers/ppov2/mod/ppov2_ab_controller.py -e AntEnv-v2 -t 600000000 -f /home/sschoepp/scratch/openai/data/PPOv2_Ant-v2:600000000_lr:0.000123_lrd:True_slrd:0.25_g:0.9839_ns:2471_mbs:1024_epo:5_eps:0.3_c1:1.0_c2:0.0019_cvl:False_mgn:0.5_gae:True_lam:0.911_hd:64_lstd:0.0_tef:3000000_ee:10_tmsf:6000000_d:cpu_ps:True_pss:33/seed$s -tl 7 --resumable -cm" >> ppo_v2b.txt
done

for s in {17..23}
do
  echo "python /home/sschoepp/scratch/openai/controllers/ppov2/mod/ppov2_ab_controller.py -e AntEnv-v2 -t 600000000 -f /home/sschoepp/scratch/openai/data/PPOv2_Ant-v2:600000000_lr:0.000123_lrd:True_slrd:0.25_g:0.9839_ns:2471_mbs:1024_epo:5_eps:0.3_c1:1.0_c2:0.0019_cvl:False_mgn:0.5_gae:True_lam:0.911_hd:64_lstd:0.0_tef:3000000_ee:10_tmsf:6000000_d:cpu_ps:True_pss:33/seed$s -tl 7 --resumable -rn" >> ppo_v2b.txt
done

for s in {17..23}
do
  echo "python /home/sschoepp/scratch/openai/controllers/ppov2/mod/ppov2_ab_controller.py -e AntEnv-v2 -t 600000000 -f /home/sschoepp/scratch/openai/data/PPOv2_Ant-v2:600000000_lr:0.000123_lrd:True_slrd:0.25_g:0.9839_ns:2471_mbs:1024_epo:5_eps:0.3_c1:1.0_c2:0.0019_cvl:False_mgn:0.5_gae:True_lam:0.911_hd:64_lstd:0.0_tef:3000000_ee:10_tmsf:6000000_d:cpu_ps:True_pss:33/seed$s -tl 7 --resumable -cm -rn" >> ppo_v2b.txt
done

echo "#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=30
#SBATCH --mem=191000M
#SBATCH --time=7-00:00
#SBATCH --job-name=ppo_v3b
#SBATCH --output=%x-%j.out
#SBATCH --mail-user=sschoepp@ualberta.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL

parallel < ppo_v3b.txt" > ppo_v3b.sh

echo "" > ppo_v3b.txt

for s in {17..23}
do
  echo "python /home/sschoepp/scratch/openai/controllers/ppov2/mod/ppov2_ab_controller.py -e AntEnv-v3 -t 600000000 -f /home/sschoepp/scratch/openai/data/PPOv2_Ant-v2:600000000_lr:0.000123_lrd:True_slrd:0.25_g:0.9839_ns:2471_mbs:1024_epo:5_eps:0.3_c1:1.0_c2:0.0019_cvl:False_mgn:0.5_gae:True_lam:0.911_hd:64_lstd:0.0_tef:3000000_ee:10_tmsf:6000000_d:cpu_ps:True_pss:33/seed$s -tl 7 --resumable" >> ppo_v3b.txt
done

for s in {17..23}
do
  echo "python /home/sschoepp/scratch/openai/controllers/ppov2/mod/ppov2_ab_controller.py -e AntEnv-v3 -t 600000000 -f /home/sschoepp/scratch/openai/data/PPOv2_Ant-v2:600000000_lr:0.000123_lrd:True_slrd:0.25_g:0.9839_ns:2471_mbs:1024_epo:5_eps:0.3_c1:1.0_c2:0.0019_cvl:False_mgn:0.5_gae:True_lam:0.911_hd:64_lstd:0.0_tef:3000000_ee:10_tmsf:6000000_d:cpu_ps:True_pss:33/seed$s -tl 7 --resumable -cm" >> ppo_v3b.txt
done

for s in {17..23}
do
  echo "python /home/sschoepp/scratch/openai/controllers/ppov2/mod/ppov2_ab_controller.py -e AntEnv-v3 -t 600000000 -f /home/sschoepp/scratch/openai/data/PPOv2_Ant-v2:600000000_lr:0.000123_lrd:True_slrd:0.25_g:0.9839_ns:2471_mbs:1024_epo:5_eps:0.3_c1:1.0_c2:0.0019_cvl:False_mgn:0.5_gae:True_lam:0.911_hd:64_lstd:0.0_tef:3000000_ee:10_tmsf:6000000_d:cpu_ps:True_pss:33/seed$s -tl 7 --resumable -rn" >> ppo_v3b.txt
done

for s in {17..23}
do
  echo "python /home/sschoepp/scratch/openai/controllers/ppov2/mod/ppov2_ab_controller.py -e AntEnv-v3 -t 600000000 -f /home/sschoepp/scratch/openai/data/PPOv2_Ant-v2:600000000_lr:0.000123_lrd:True_slrd:0.25_g:0.9839_ns:2471_mbs:1024_epo:5_eps:0.3_c1:1.0_c2:0.0019_cvl:False_mgn:0.5_gae:True_lam:0.911_hd:64_lstd:0.0_tef:3000000_ee:10_tmsf:6000000_d:cpu_ps:True_pss:33/seed$s -tl 7 --resumable -cm -rn" >> ppo_v3b.txt
done

echo "#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=30
#SBATCH --mem=191000M
#SBATCH --time=7-00:00
#SBATCH --job-name=ppo_v4b
#SBATCH --output=%x-%j.out
#SBATCH --mail-user=sschoepp@ualberta.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL

parallel < ppo_v4b.txt" > ppo_v4b.sh

echo "" > ppo_v4b.txt

for s in {17..23}
do
  echo "python /home/sschoepp/scratch/openai/controllers/ppov2/mod/ppov2_ab_controller.py -e AntEnv-v3 -t 600000000 -f /home/sschoepp/scratch/openai/data/PPOv2_Ant-v2:600000000_lr:0.000123_lrd:True_slrd:0.25_g:0.9839_ns:2471_mbs:1024_epo:5_eps:0.3_c1:1.0_c2:0.0019_cvl:False_mgn:0.5_gae:True_lam:0.911_hd:64_lstd:0.0_tef:3000000_ee:10_tmsf:6000000_d:cpu_ps:True_pss:33/seed$s -tl 7 --resumable" >> ppo_v4b.txt
done

for s in {17..23}
do
  echo "python /home/sschoepp/scratch/openai/controllers/ppov2/mod/ppov2_ab_controller.py -e AntEnv-v3 -t 600000000 -f /home/sschoepp/scratch/openai/data/PPOv2_Ant-v2:600000000_lr:0.000123_lrd:True_slrd:0.25_g:0.9839_ns:2471_mbs:1024_epo:5_eps:0.3_c1:1.0_c2:0.0019_cvl:False_mgn:0.5_gae:True_lam:0.911_hd:64_lstd:0.0_tef:3000000_ee:10_tmsf:6000000_d:cpu_ps:True_pss:33/seed$s -tl 7 --resumable -cm" >> ppo_v4b.txt
done

for s in {17..23}
do
  echo "python /home/sschoepp/scratch/openai/controllers/ppov2/mod/ppov2_ab_controller.py -e AntEnv-v3 -t 600000000 -f /home/sschoepp/scratch/openai/data/PPOv2_Ant-v2:600000000_lr:0.000123_lrd:True_slrd:0.25_g:0.9839_ns:2471_mbs:1024_epo:5_eps:0.3_c1:1.0_c2:0.0019_cvl:False_mgn:0.5_gae:True_lam:0.911_hd:64_lstd:0.0_tef:3000000_ee:10_tmsf:6000000_d:cpu_ps:True_pss:33/seed$s -tl 7 --resumable -rn" >> ppo_v4b.txt
done

for s in {17..23}
do
  echo "python /home/sschoepp/scratch/openai/controllers/ppov2/mod/ppov2_ab_controller.py -e AntEnv-v3 -t 600000000 -f /home/sschoepp/scratch/openai/data/PPOv2_Ant-v2:600000000_lr:0.000123_lrd:True_slrd:0.25_g:0.9839_ns:2471_mbs:1024_epo:5_eps:0.3_c1:1.0_c2:0.0019_cvl:False_mgn:0.5_gae:True_lam:0.911_hd:64_lstd:0.0_tef:3000000_ee:10_tmsf:6000000_d:cpu_ps:True_pss:33/seed$s -tl 7 --resumable -cm -rn" >> ppo_v4b.txt
done