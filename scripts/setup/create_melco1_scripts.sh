echo 'PPO_AB_CONTROLLER_ABSOLUTE_PATH="/home/sschoepp/Documents/openai/controllers/ppov2/mod2/ppov2_ab_controller.py"' > melco1.sh
echo 'RESUME_FILE="/local/melco2/shared/PPOv2_AntEnv-v3:600000000_Ant-v2:600000000_lr:0.000123_lrd:True_slrd:0.25_g:0.9839_ns:2471_mbs:1024_epo:5_eps:0.3_c1:1.0_c2:0.0019_cvl:False_mgn:0.5_gae:True_lam:0.911_hd:64_lstd:0.0_tef:3000000_ee:10_tmsf:50000000_cm:False_rn:False_d:cpu_r"' >> melco1.sh

for s in {0..9}
do
  echo 'tmux new-session -d -s ppov3-'$s' "python $PPO_AB_CONTROLLER_ABSOLUTE_PATH --resume -rf $RESUME_FILE/seed'$s'"' >> melco1.sh
done

echo 'RESUME_FILE="/local/melco2/shared/PPOv2_AntEnv-v3:600000000_Ant-v2:600000000_lr:0.000123_lrd:True_slrd:0.25_g:0.9839_ns:2471_mbs:1024_epo:5_eps:0.3_c1:1.0_c2:0.0019_cvl:False_mgn:0.5_gae:True_lam:0.911_hd:64_lstd:0.0_tef:3000000_ee:10_tmsf:50000000_cm:True_rn:False_d:cpu_r"' >> melco1.sh

for s in {0..3}  # todo: 4 - 9
do
  echo 'tmux new-session -d -s ppov3cm-'$s' "python $PPO_AB_CONTROLLER_ABSOLUTE_PATH --resume -rf $RESUME_FILE/seed'$s'"' >> melco1.sh
done

#echo 'RESUME_FILE="/local/melco2/shared/PPOv2_AntEnv-v3:600000000_Ant-v2:600000000_lr:0.000123_lrd:True_slrd:0.25_g:0.9839_ns:2471_mbs:1024_epo:5_eps:0.3_c1:1.0_c2:0.0019_cvl:False_mgn:0.5_gae:True_lam:0.911_hd:64_lstd:0.0_tef:3000000_ee:10_tmsf:50000000_cm:False_rn:True_d:cpu_r"' >> melco1.sh
#
#for s in {0..9}
#do
#  echo 'tmux new-session -d -s ppov3rn-'$s' "python $PPO_AB_CONTROLLER_ABSOLUTE_PATH --resume -rf $RESUME_FILE/seed'$s'"' >> melco1.sh
#done
#
#echo 'RESUME_FILE="/local/melco2/shared/PPOv2_AntEnv-v3:600000000_Ant-v2:600000000_lr:0.000123_lrd:True_slrd:0.25_g:0.9839_ns:2471_mbs:1024_epo:5_eps:0.3_c1:1.0_c2:0.0019_cvl:False_mgn:0.5_gae:True_lam:0.911_hd:64_lstd:0.0_tef:3000000_ee:10_tmsf:50000000_cm:True_rn:True_d:cpu_r"' >> melco1.sh
#
#for s in {0..9}
#do
#  echo 'tmux new-session -d -s ppov3cmrn-'$s' "python $PPO_AB_CONTROLLER_ABSOLUTE_PATH --resume -rf $RESUME_FILE/seed'$s'"' >> melco1.sh
#done
