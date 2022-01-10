PPO_AB_CONTROLLER_ABSOLUTE_PATH="/home/sschoepp/Documents/openai/controllers/ppov2/mod2/ppov2_ab_controller.py"
#FILE="/home/sschoepp/Documents/openai/data/PPOv2_Ant-v2:600000000_lr:0.000123_lrd:True_slrd:0.25_g:0.9839_ns:2471_mbs:1024_epo:5_eps:0.3_c1:1.0_c2:0.0019_cvl:False_mgn:0.5_gae:True_lam:0.911_hd:64_lstd:0.0_tef:3000000_ee:10_tmsf:50000000_d:cpu_ps:True_pss:33_resumed"
FILE="/home/sschoepp/Documents/openai/data/PPOv2_Ant-v2:600000000_lr:0.000123_lrd:True_slrd:0.25_g:0.9839_ns:2471_mbs:1024_epo:5_eps:0.3_c1:1.0_c2:0.0019_cvl:False_mgn:0.5_gae:True_lam:0.911_hd:64_lstd:0.0_tef:3000000_ee:10_tmsf:6000000_d:cpu_ps:True_pss:33"

for s in {10..29}
do
  tmux new-session -d -s ppov1-$s "python $PPO_AB_CONTROLLER_ABSOLUTE_PATH -e AntEnv-v1 -t 3000000 -f $FILE/seed$s; python $PPO_AB_CONTROLLER_ABSOLUTE_PATH -e AntEnv-v3 -t 3000000 -f $FILE/seed$s"
  tmux new-session -d -s ppov2-$s "python $PPO_AB_CONTROLLER_ABSOLUTE_PATH -e AntEnv-v2 -t 3000000 -f $FILE/seed$s; python $PPO_AB_CONTROLLER_ABSOLUTE_PATH -e AntEnv-v4 -t 3000000 -f $FILE/seed$s"
done