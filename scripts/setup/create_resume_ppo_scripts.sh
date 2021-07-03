echo 'PPO_N_CONTROLLER_ABSOLUTE_PATH="/home/sschoepp/Documents/openai/controllers/ppov2/ppov2_n_controller.py"' > melco2.sh

echo 'RESUME_FILE="/home/sschoepp/Documents/openai/data/PPOv2_Ant-v2:1000000000_lr:0.000348_lrd:True_g:0.8944_ns:3648_mbs:128_epo:6_eps:0.1_c1:0.5_c2:0.0037_cvl:False_mgn:0.5_gae:True_lam:0.9049_hd:64_lstd:0.0_tef:5000000_ee:10_tmsf:50000000_d:cpu_ps:True_pss:45"' >> melco2.sh

for s in {0..9}
do
  echo 'tmux new-session -d -s ppo45'$s' "python $PPO_N_CONTROLLER_ABSOLUTE_PATH --resume --resume_file $RESUME_FILE/seed'$s'"' >> melco2.sh
done