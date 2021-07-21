echo 'SAC_N_CONTROLLER_ABSOLUTE_PATH="/home/sschoepp/Documents/openai/controllers/sacv2/sacv2_n_controller.py"' > melco2.sh
echo 'RESUME_FILE="/local/melco2-1/shared/ant/faulty/v1/SACv2_AntEnv-v1:20000000_Ant-v2:20000000_g:0.9646_t:0.0877_a:0.2_lr:0.001092_hd:256_rbs:500000_bs:512_mups:1_tui:1_tef:100000_ee:10_tmsf:1000000_crb:False_rn:False_a:True_d:cuda_r"' >> melco2.sh

for i in {0..4}
do
  echo 'tmux new-session -d -s sac'$i' "CUDA_VISIBLE_DEVICES=0 python $SAC_N_CONTROLLER_ABSOLUTE_PATH --resume --resume_file $RESUME_FILE/seed'$i'"' >> melco2.sh
done

for i in {5..9}
do
  echo 'tmux new-session -d -s sac'$i' "CUDA_VISIBLE_DEVICES=1 python $SAC_N_CONTROLLER_ABSOLUTE_PATH --resume --resume_file $RESUME_FILE/seed'$i'"' >> melco2.sh
done