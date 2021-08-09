#SAC_N_CONTROLLER_ABSOLUTE_PATH="/home/sschoepp/Documents/openai/controllers/sacv2/mod/sacv2_n_controller.py"

#for s in 14 15 16 17
#do
#  tmux new-session -d -s sac-$s "CUDA_VISIBLE_DEVICES=6 python $SAC_N_CONTROLLER_ABSOLUTE_PATH -e FetchReachEnv-v0 -a -c -t 2000000 -tef 10000 -tmsf 20000 -ps -pss 21 -s $s"
#done
#
#for s in 18 19 20 21
#do
#  tmux new-session -d -s sac-$s "CUDA_VISIBLE_DEVICES=7 python $SAC_N_CONTROLLER_ABSOLUTE_PATH -e FetchReachEnv-v0 -a -c -t 2000000 -tef 10000 -tmsf 20000 -ps -pss 21 -s $s"
#done

#for s in 22 23 24 25 26 27 28
#do
#  tmux new-session -d -s sac-$s "CUDA_VISIBLE_DEVICES=5 python $SAC_N_CONTROLLER_ABSOLUTE_PATH -e FetchReachEnv-v0 -a -c -t 2000000 -tef 10000 -tmsf 20000 -ps -pss 21 -s $s"
#done

SAC_AB_CONTROLLER_ABSOLUTE_PATH="/home/sschoepp/Documents/openai/controllers/sacv2/mod/sacv2_ab_controller.py"
FILE="/home/sschoepp/Documents/openai/data/SACv2_FetchReachEnv-v0:2000000_g:0.8097_t:0.0721_a:0.2_lr:0.001738_hd:256_rbs:10000_bs:512_mups:1_tui:1_tef:10000_ee:10_tmsf:20000_a:True_d:cuda_ps:True_pss:21_mod"

for s in {10..15}
do
  tmux new-session -d -s saccrbrn-$s "CUDA_VISIBLE_DEVICES=3 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -c -crb -rn -e FetchReachEnv-v1 -f $FILE/seed$s -t 2000000"
done

for s in {16..21}
do
  tmux new-session -d -s saccrbrn-$s "CUDA_VISIBLE_DEVICES=4 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -c -crb -rn -e FetchReachEnv-v1 -f $FILE/seed$s -t 2000000"
done

for s in {22..27}
do
  tmux new-session -d -s saccrbrn-$s "CUDA_VISIBLE_DEVICES=5 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -c -crb -rn -e FetchReachEnv-v1 -f $FILE/seed$s -t 2000000"
done

for s in {28..29}
do
  tmux new-session -d -s saccrbrn-$s "CUDA_VISIBLE_DEVICES=6 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -c -crb -rn -e FetchReachEnv-v1 -f $FILE/seed$s -t 2000000"
done

tmux new-session -d -s saccrbrnv1-19 "CUDA_VISIBLE_DEVICES=6 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -c -e FetchReachEnv-v1 -f $FILE/seed19 -t 2000000"
tmux new-session -d -s saccrbrnv1-12 "CUDA_VISIBLE_DEVICES=6 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -c -crb -e FetchReachEnv-v1 -f $FILE/seed12 -t 2000000"
tmux new-session -d -s saccrbrnv1-27 "CUDA_VISIBLE_DEVICES=6 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -c -crb -e FetchReachEnv-v1 -f $FILE/seed27 -t 2000000"
tmux new-session -d -s saccrbrnv1-20 "CUDA_VISIBLE_DEVICES=6 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -c -rn -e FetchReachEnv-v1 -f $FILE/seed19 -t 2000000"

#tmux new-session -d -s saccrbrn4-19 "CUDA_VISIBLE_DEVICES=0 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -c -rn -e FetchReachEnv-v4 -f $FILE/seed19 -t 2000000"
#tmux new-session -d -s saccrbrn4-12 "CUDA_VISIBLE_DEVICES=1 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -c -rn -e FetchReachEnv-v4 -f $FILE/seed19 -t 2000000"
#tmux new-session -d -s saccrbrn4-27 "CUDA_VISIBLE_DEVICES=3 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -c -rn -e FetchReachEnv-v4 -f $FILE/seed19 -t 2000000"
#tmux new-session -d -s saccrbrn4-20 "CUDA_VISIBLE_DEVICES=4 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -c -rn -e FetchReachEnv-v4 -f $FILE/seed19 -t 2000000"