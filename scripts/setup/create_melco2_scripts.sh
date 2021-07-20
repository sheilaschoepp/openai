echo 'SAC_AB_CONTROLLER_ABSOLUTE_PATH="/home/sschoepp/Documents/openai/controllers/sacv2/sacv2_ab_controller.py"' > melco2.sh
echo 'RESUME_FILE="/local/melco2-1/shared/fetchreach/normal/SACv2_FetchReachEnv-v0:2000000_g:0.8097_t:0.0721_a:0.2_lr:0.001738_hd:256_rbs:10000_bs:512_mups:1_tui:1_tef:10000_ee:10_tmsf:20000_a:True_d:cuda_ps:True_pss:21"' >> melco2.sh

for s in {0..4}
do
  echo 'tmux new-session -d -s sac'$s' "CUDA_VISIBLE_DEVICES=0 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -e FetchReachEnv-v1 -t 2000000 -c -f $RESUME_FILE/seed'$s'"' >> melco2.sh
done
for s in {5..9}
do
  echo 'tmux new-session -d -s sac'$s' "CUDA_VISIBLE_DEVICES=1 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -e FetchReachEnv-v1 -t 2000000 -c -f $RESUME_FILE/seed'$s'"' >> melco2.sh
done



echo 'PPO_AB_CONTROLLER_ABSOLUTE_PATH="/home/sschoepp/Documents/openai/controllers/ppov2/ppov2_ab_controller.py"' >> melco2.sh
echo 'RESUME_FILE="/local/melco2-1/shared/fetchreach/normal/PPOv2_FetchReachEnv-v0:6000000_lr:0.000275_lrd:True_g:0.848_ns:3424_mbs:8_epo:24_eps:0.3_c1:1.0_c2:0.0007_cvl:False_mgn:0.5_gae:True_lam:0.9327_hd:64_lstd:0.0_tef:30000_ee:10_tmsf:60000_d:cpu_ps:True_pss:43"' >> melco2.sh

for s in {0..9}
do
  echo 'tmux new-session -d -s ppov1-'$i' "python $PPO_AB_CONTROLLER_ABSOLUTE_PATH -e FetchReachEnv-v1 -t 6000000 -f $RESUME_FILE/seed'$s'"' >> melco2.sh
done

for s in {0..9}
do
  echo 'tmux new-session -d -s ppov1cm-'$i' "python $PPO_AB_CONTROLLER_ABSOLUTE_PATH -e FetchReachEnv-v1 -t 6000000 -f $RESUME_FILE/seed'$s' -cm"' >> melco2.sh
done

for s in {0..9}
do
  echo 'tmux new-session -d -s ppov1rn-'$i' "python $PPO_AB_CONTROLLER_ABSOLUTE_PATH -e FetchReachEnv-v1 -t 6000000 -f $RESUME_FILE/seed'$s' -rn"' >> melco2.sh
done

for s in {0..9}
do
  echo 'tmux new-session -d -s ppov1cmrn-'$i' "python $PPO_AB_CONTROLLER_ABSOLUTE_PATH -e FetchReachEnv-v1 -t 6000000 -f $RESUME_FILE/seed'$s' -cm -rn"' >> melco2.sh
done

for s in {0..9}
do
  echo 'tmux new-session -d -s ppov4-'$i' "python $PPO_AB_CONTROLLER_ABSOLUTE_PATH -e FetchReachEnv-v4 -t 6000000 -f $RESUME_FILE/seed'$s'"' >> melco2.sh
done