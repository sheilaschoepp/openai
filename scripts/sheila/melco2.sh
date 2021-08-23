SAC_AB_CONTROLLER_ABSOLUTE_PATH="/home/sschoepp/Documents/openai/controllers/sacv2/mod/sacv2_ab_controller.py"
FILE="/local/melco2-1/shared/ant/seeds/normal/SACv2_Ant-v2:20000000_g:0.9646_t:0.0877_a:0.2_lr:0.001092_hd:256_rbs:500000_bs:512_mups:1_tui:1_tef:100000_ee:10_tmsf:100000_a:True_d:cuda_ps:True_pss:61_resumed_mod"
for s in 17 18 19
do
  tmux new-session -d -s sac-$s "CUDA_VISIBLE_DEVICES=0 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -s $s -c -e AntEnv-v1 -f $FILE/seed$s -t 200000000 --resumable -rn"
done

for s in 17 18 19
do
  tmux new-session -d -s sac2-$s "CUDA_VISIBLE_DEVICES=1 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -s $s -c -e AntEnv-v1 -f $FILE/seed$s -t 20000000 --resumable -crb -rn"
done
