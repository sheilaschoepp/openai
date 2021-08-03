SAC_AB_CONTROLLER_ABSOLUTE_PATH="/home/sschoepp/Documents/openai/controllers/sacv2/sacv2_ab_controller.py"
RESUME_FILE="/home/sschoepp/Documents/openai/data/SACv2_AntEnv-v2:20000000_Ant-v2:20000000_g:0.9646_t:0.0877_a:0.2_lr:0.001092_hd:256_rbs:500000_bs:512_mups:1_tui:1_tef:100000_ee:10_tmsf:1000000_crb:True_rn:False_a:True_d:cuda_r"

for s in {0..4}
do
  tmux new-session -d -s sac-$s "CUDA_VISIBLE_DEVICES=6 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -c --resume -rf $RESUME_FILE/seed$s"
done
for s in {5..9}
do
  tmux new-session -d -s sac-$s "CUDA_VISIBLE_DEVICES=7 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -c --resume -rf $RESUME_FILE/seed$s"
done