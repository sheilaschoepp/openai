echo 'SAC_AB_CONTROLLER_ABSOLUTE_PATH="/home/sschoepp/Documents/openai/controllers/sacv2/sacv2_ab_controller.py"' > melco1.sh
echo 'RESUME_FILE="/local/melco2/shared/SACv2_FetchReachEnv-v0:2000000_g:0.8097_t:0.0721_a:0.2_lr:0.001738_hd:256_rbs:10000_bs:512_mups:1_tui:1_tef:10000_ee:10_tmsf:20000_a:True_d:cuda_ps:True_pss:21"' >> melco1.sh

for s in {0..4}
do
  echo 'tmux new-session -d -s sacv4crb-'$s' "CUDA_VISIBLE_DEVICES=0 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -e FetchReachEnv-v4 -t 2000000 -c -f $RESUME_FILE/seed'$s' -crb"' >> melco1.sh
done
for s in {5..9}
do
  echo 'tmux new-session -d -s sacv4crb-'$s' "CUDA_VISIBLE_DEVICES=1 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -e FetchReachEnv-v4 -t 2000000 -c -f $RESUME_FILE/seed'$s' -crb"' >> melco1.sh
done

for s in {0..4}
do
  echo 'tmux new-session -d -s sacv4rn-'$s' "CUDA_VISIBLE_DEVICES=3 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -e FetchReachEnv-v4 -t 2000000 -c -f $RESUME_FILE/seed'$s' -rn"' >> melco1.sh
done
for s in {5..9}
do
  echo 'tmux new-session -d -s sacv4rn-'$s' "CUDA_VISIBLE_DEVICES=4 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -e FetchReachEnv-v4 -t 2000000 -c -f $RESUME_FILE/seed'$s' -rn"' >> melco1.sh
done

for s in {0..4}
do
  echo 'tmux new-session -d -s sacv4crbrn-'$s' "CUDA_VISIBLE_DEVICES=6 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -e FetchReachEnv-v4 -t 2000000 -c -f $RESUME_FILE/seed'$s' -crb -rn"' >> melco1.sh
done
for s in {5..9}
do
  echo 'tmux new-session -d -s sacv4crbrn-'$s' "CUDA_VISIBLE_DEVICES=7 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -e FetchReachEnv-v4 -t 2000000 -c -f $RESUME_FILE/seed'$s' -crb -rn"' >> melco1.sh
done
