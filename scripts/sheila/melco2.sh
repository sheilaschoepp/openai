#PPO_AB_CONTROLLER_ABSOLUTE_PATH="/home/sschoepp/Documents/openai/controllers/ppov2/mod3/ppov2_ab_controller.py"
#FILE="/local/melco2-1/shared/fetchreach/seeds/normal/PPOv2_FetchReachEnv-v0:6000000_lr:0.000275_lrd:True_g:0.848_ns:3424_mbs:8_epo:24_eps:0.3_c1:1.0_c2:0.0007_cvl:False_mgn:0.5_gae:True_lam:0.9327_hd:64_lstd:0.0_tef:30000_ee:10_tmsf:60000_d:cpu_ps:True_pss:43"
#
#for s in {0..1}
#do
#  tmux new-session -d -s ppov1cmrn-$s "python $PPO_AB_CONTROLLER_ABSOLUTE_PATH -e FetchReachEnv-v4 -t 6000000 -f $FILE/seed$s -cm -rn"
#  tmux new-session -d -s ppov1-$s "python $PPO_AB_CONTROLLER_ABSOLUTE_PATH -e FetchReachEnv-v4 -t 6000000 -f $FILE/seed$s"
#  tmux new-session -d -s ppov2cmrn-$s "python $PPO_AB_CONTROLLER_ABSOLUTE_PATH -e FetchReachEnv-v6 -t 6000000 -f $FILE/seed$s -cm -rn"
#  tmux new-session -d -s ppov2-$s "python $PPO_AB_CONTROLLER_ABSOLUTE_PATH -e FetchReachEnv-v6 -t 6000000 -f $FILE/seed$s"
#done

SAC_AB_CONTROLLER_ABSOLUTE_PATH="/home/sschoepp/Documents/openai/controllers/sacv2/mod2/sacv2_ab_controller.py"
FILE="/local/melco2-1/shared/fetchreach/seeds/normal/SACv2_FetchReachEnv-v0:2000000_g:0.8097_t:0.0721_a:0.2_lr:0.001738_hd:256_rbs:10000_bs:512_mups:1_tui:1_tef:10000_ee:10_tmsf:20000_a:True_d:cuda_ps:True_pss:21_mod"

for s in {0..1}
do
  tmux new-session -d -s sacv1crbrn-$s "CUDA_VISIBLE_DEVICES=0 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -e FetchReachEnv-v4 -t 50000 -f $FILE/seed$s -cm -rn"
  tmux new-session -d -s sacv1-$s "CUDA_VISIBLE_DEVICES=0 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -e FetchReachEnv-v4 -t 50000 -f $FILE/seed$s"
  tmux new-session -d -s sacv2crbrn-$s "CUDA_VISIBLE_DEVICES=1 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -e FetchReachEnv-v6 -t 50000 -f $FILE/seed$s -cm -rn"
  tmux new-session -d -s sacv2-$s "CUDA_VISIBLE_DEVICES=1 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -e FetchReachEnv-v6 -t 50000 -f $FILE/seed$s"
done