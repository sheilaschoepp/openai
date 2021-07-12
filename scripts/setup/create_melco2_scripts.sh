echo 'SAC_N_CONTROLLER_ABSOLUTE_PATH="/home/sschoepp/Documents/openai/controllers/sacv2/sacv2_n_controller.py"' > melco2.sh
echo 'RESUME_FILE="/local/melco2-1/shared/ant/normal/SACv2_Ant-v2:20000000_g:0.9646_t:0.0877_a:0.2_lr:0.001092_hd:256_rbs:500000_bs:512_mups:1_tui:1_tef:100000_ee:10_tmsf:1000000_a:True_d:cuda_ps:True_pss:61_resumed"' >> melco2.sh
for s in {0..4}
do
  echo 'tmux new-session -d -s sac61'$s' "CUDA_VISIBLE_DEVICES=0 python $SAC_N_CONTROLLER_ABSOLUTE_PATH -e AntEnv-v3 --resume --resume_file $RESUME_FILE/seed'$s' -t 20000000 -tl 4"' >> melco2.sh
done
for s in {5..9}
do
  echo 'tmux new-session -d -s sac61'$s' "CUDA_VISIBLE_DEVICES=1 python $SAC_N_CONTROLLER_ABSOLUTE_PATH -e AntEnv-v3 --resume --resume_file $RESUME_FILE/seed'$s' -t 20000000 -tl 4"' >> melco2.sh
done

for s in {0..4}
do
  echo 'tmux new-session -d -s sac61'$s'crb "CUDA_VISIBLE_DEVICES=3 python $SAC_N_CONTROLLER_ABSOLUTE_PATH -crb -e AntEnv-v3 --resume --resume_file $RESUME_FILE/seed'$s' -t 20000000 -tl 4"' >> melco2.sh
done
for s in {5..9}
do
  echo 'tmux new-session -d -s sac61'$s'crb "CUDA_VISIBLE_DEVICES=4 python $SAC_N_CONTROLLER_ABSOLUTE_PATH -crb -e AntEnv-v3 --resume --resume_file $RESUME_FILE/seed'$s' -t 20000000 -tl 4"' >> melco2.sh
done

for s in {0..4}
do
  echo 'tmux new-session -d -s sac61'$s'rn "CUDA_VISIBLE_DEVICES=6 python $SAC_N_CONTROLLER_ABSOLUTE_PATH -rn -e AntEnv-v3 --resume --resume_file $RESUME_FILE/seed'$s' -t 20000000 -tl 4"' >> melco2.sh
done
#for s in {5..9} --resume
#do
#  echo 'tmux new-session -d -s sac61'$s'rn "CUDA_VISIBLE_DEVICES=7 python $SAC_N_CONTROLLER_ABSOLUTE_PATH -rn -e AntEnv-v3 --resume --resume_file $RESUME_FILE/seed'$s' -t 20000000 -tl 4"' >> melco2.sh
#done