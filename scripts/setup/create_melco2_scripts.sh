#echo 'SAC_AB_CONTROLLER_ABSOLUTE_PATH="/home/sschoepp/Documents/openai/controllers/sacv2/sacv2_ab_controller.py"' > melco2.sh
#echo 'RESUME_FILE="/local/melco2-1/shared/ant/normal/SACv2_Ant-v2:20000000_g:0.9646_t:0.0877_a:0.2_lr:0.001092_hd:256_rbs:500000_bs:512_mups:1_tui:1_tef:100000_ee:10_tmsf:1000000_a:True_d:cuda_ps:True_pss:61_resumed"' >> melco2.sh

#for s in {0..4}
#do
#  echo 'tmux new-session -d -s sac61'$s' "CUDA_VISIBLE_DEVICES=0 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -e AntEnv-v2 -t 20000000 -c -f $RESUME_FILE/seed'$s' --resumable"' >> melco2.sh
#done
#for s in {5..9}
#do
#  echo 'tmux new-session -d -s sac61'$s' "CUDA_VISIBLE_DEVICES=1 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -e AntEnv-v2 -t 20000000 -c -f $RESUME_FILE/seed'$s' --resumable"' >> melco2.sh
#done

#for s in {0..4}
#do
#  echo 'tmux new-session -d -s sac61'$s'crb "CUDA_VISIBLE_DEVICES=3 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -e AntEnv-v2 -t 20000000 -c -f $RESUME_FILE/seed'$s' --resumable -crb"' >> melco2.sh
#done
#for s in {5..9}
#do
#  echo 'tmux new-session -d -s sac61'$s'crb "CUDA_VISIBLE_DEVICES=4 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -e AntEnv-v2 -t 20000000 -c -f $RESUME_FILE/seed'$s' --resumable -crb"' >> melco2.sh
#done
#
#for s in {0..4}
#do
#  echo 'tmux new-session -d -s sac61'$s'rn "CUDA_VISIBLE_DEVICES=6 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -e AntEnv-v2 -t 20000000 -c -f $RESUME_FILE/seed'$s' --resumable -rn"' >> melco2.sh
#done
#for s in {5..9}
#do
#  echo 'tmux new-session -d -s sac61'$s'rn "CUDA_VISIBLE_DEVICES=7 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -e AntEnv-v2 -t 20000000 -c -f $RESUME_FILE/seed'$s' --resumable -rn"' >> melco2.sh
#done
#
#for s in {0..4}
#do
#  echo 'tmux new-session -d -s sac61'$s'crbrn "CUDA_VISIBLE_DEVICES=0 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -e AntEnv-v2 -t 20000000 -c -f $RESUME_FILE/seed'$s' --resumable -crb -rn"' >> melco2.sh
#done
#for s in {5..9}
#do
#  echo 'tmux new-session -d -s sac61'$s'crbrn "CUDA_VISIBLE_DEVICES=1 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -e AntEnv-v2 -t 20000000 -c -f $RESUME_FILE/seed'$s' --resumable -crb -rn"' >> melco2.sh
#done

#echo 'SAC_N_CONTROLLER_ABSOLUTE_PATH="/home/sschoepp/Documents/openai/controllers/sacv2/sacv2_n_controller.py"' > melco2.sh
#
#for s in {10..14}
#do
#  echo 'tmux new-session -d -s sac61'$s' "CUDA_VISIBLE_DEVICES=0 python $SAC_N_CONTROLLER_ABSOLUTE_PATH -a -c -s '$s' --resumable -ps -pss 61"' >> melco2.sh
#done

echo 'PPO_N_CONTROLLER_ABSOLUTE_PATH="/home/sschoepp/Documents/openai/controllers/ppov2/mod2/ppov2_n_controller.py"' > melco2.sh

for i in {0..9}
do
  echo 'tmux new-session -d -s ppo33-25'$i' "python $PPO_N_CONTROLLER_ABSOLUTE_PATH --resume -rf /local/melco2-1/shared/ant/lr/PPOv2_Ant-v2:600000000_lr:0.000123_lrd:True_slrd:0.25_g:0.9839_ns:2471_mbs:1024_epo:5_eps:0.3_c1:1.0_c2:0.0019_cvl:False_mgn:0.5_gae:True_lam:0.911_hd:64_lstd:0.0_tef:3000000_ee:10_tmsf:50000000_d:cpu_ps:True_pss:33_r/seed'$i'"' >> melco2.sh
done