#SAC_AB_CONTROLLER_ABSOLUTE_PATH="/home/sschoepp/Documents/openai/controllers/sacv2/mod/sacv2_ab_controller.py"
#FILE="/local/melco2-1/shared/ant/seeds/normal/SACv2_Ant-v2:20000000_g:0.9646_t:0.0877_a:0.2_lr:0.001092_hd:256_rbs:500000_bs:512_mups:1_tui:1_tef:100000_ee:10_tmsf:100000_a:True_d:cuda_ps:True_pss:61_resumed_mod"
#
#for s in 17 18 19
#do
#  tmux new-session -d -s sac1-$s "CUDA_VISIBLE_DEVICES=0 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -c -e AntEnv-v1 -f $FILE/seed$s -t 200000000 --resumable -rn"
#done
#
#for s in 17 18 19
#do
#  tmux new-session -d -s sac2-$s "CUDA_VISIBLE_DEVICES=1 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -c -e AntEnv-v1 -f $FILE/seed$s -t 20000000 --resumable -crb -rn"
#done
#for s in 17 18 19
#do
#  tmux new-session -d -s sac3-$s "CUDA_VISIBLE_DEVICES=0 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -c -e AntEnv-v3 -f $FILE/seed$s -t 200000000 --resumable -rn"
#done
#
#for s in 17 18 19
#do
#  tmux new-session -d -s sac4-$s "CUDA_VISIBLE_DEVICES=1 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -c -e AntEnv-v3 -f $FILE/seed$s -t 20000000 --resumable -crb -rn"
#done


PPO_AB_CONTROLLER_ABSOLUTE_PATH="/home/sschoepp/Documents/openai/controllers/ppov2/mod/ppov2_ab_controller.py"
FILE="/local/melco2-1/shared/ant/seeds/faulty/ppo/v3/old/PPOv2_AntEnv-v3:600000000_Ant-v2:600000000_lr:0.000123_lrd:True_slrd:0.25_g:0.9839_ns:2471_mbs:1024_epo:5_eps:0.3_c1:1.0_c2:0.0019_cvl:False_mgn:0.5_gae:True_lam:0.911_hd:64_lstd:0.0_tef:3000000_ee:10_tmsf:6000000_cm:True_rn:True_d:cpu_r"

for s in {10..29}
do
  tmux new-session -d -s ppo-$s "python $PPO_AB_CONTROLLER_ABSOLUTE_PATH --resume -rf $FILE/seed$s"
done

FILE="/local/melco2-1/shared/ant/seeds/faulty/ppo/v3/old/PPOv2_AntEnv-v2:600000000_Ant-v2:600000000_lr:0.000123_lrd:True_slrd:0.25_g:0.9839_ns:2471_mbs:1024_epo:5_eps:0.3_c1:1.0_c2:0.0019_cvl:False_mgn:0.5_gae:True_lam:0.911_hd:64_lstd:0.0_tef:3000000_ee:10_tmsf:6000000_cm:True_rn:True_d:cpu_r"

for s in {10..29}
do
  tmux new-session -d -s ppo2-$s "python $PPO_AB_CONTROLLER_ABSOLUTE_PATH --resume -rf $FILE/seed$s"
done