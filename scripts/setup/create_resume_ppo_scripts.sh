echo 'PPO_N_CONTROLLER_ABSOLUTE_PATH="/home/sschoepp/Documents/openai/controllers/ppov2/ppov2_n_controller.py"' > melco2.sh

echo 'RESUME_FILE="/home/sschoepp/Documents/openai/data/PPOv2_Ant-v2:1000000000_lr:0.000123_lrd:True_g:0.9839_ns:2471_mbs:1024_epo:5_eps:0.3_c1:1.0_c2:0.0019_cvl:False_mgn:0.5_gae:True_lam:0.911_hd:64_lstd:0.0_tef:5000000_ee:10_tmsf:50000000_d:cpu_ps:True_pss:33"' >> melco2.sh

for s in {0..9}
do
  echo 'tmux new-session -d -s ppo33'$s' "python $PPO_N_CONTROLLER_ABSOLUTE_PATH --resume --resume_file $RESUME_FILE/seed$s"' >> melco2.sh
done