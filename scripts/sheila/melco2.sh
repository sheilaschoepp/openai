#PPO_AB_CONTROLLER_ABSOLUTE_PATH="/home/sschoepp/Documents/openai/controllers/ppov2/ppov2_ab_controller.py"
#FILE="/local/melco2-1/shared/fetchreach/seeds/normal/PPOv2_FetchReachEnv-v0:6000000_lr:0.000275_lrd:True_g:0.848_ns:3424_mbs:8_epo:24_eps:0.3_c1:1.0_c2:0.0007_cvl:False_mgn:0.5_gae:True_lam:0.9327_hd:64_lstd:0.0_tef:30000_ee:10_tmsf:60000_d:cpu_ps:True_pss:43"
#
#for s in {20..29}
#do
#  tmux new-session -d -s ppov1-$s "python $PPO_AB_CONTROLLER_ABSOLUTE_PATH -e FetchReachEnv-v1 -f $FILE/seed$s -t 6000000"
#done
#
#for s in {20..29}
#do
#  tmux new-session -d -s ppov1cm-$s "python $PPO_AB_CONTROLLER_ABSOLUTE_PATH -e FetchReachEnv-v1 -f $FILE/seed$s -t 6000000 -cm"
#done
#
#for s in {20..29}
#do
#  tmux new-session -d -s ppov1rn-$s "python $PPO_AB_CONTROLLER_ABSOLUTE_PATH -e FetchReachEnv-v1 -f $FILE/seed$s -t 6000000 -rn"
#done


#SAC_N_CONTROLLER_ABSOLUTE_PATH="/home/sschoepp/Documents/openai/controllers/sacv2/mod/sacv2_n_controller.py"
#
#for s in 10 11
#do
#  tmux new-session -d -s sac-$s "CUDA_VISIBLE_DEVICES=0 python $SAC_N_CONTROLLER_ABSOLUTE_PATH -e FetchReachEnv-v0 -a -c -t 2000000 -tef 10000 -tmsf 20000 -ps -pss 21 -s $s"
#done
#
#for s in 12 13
#do
#  tmux new-session -d -s sac-$s "CUDA_VISIBLE_DEVICES=1 python $SAC_N_CONTROLLER_ABSOLUTE_PATH -e FetchReachEnv-v0 -a -c -t 2000000 -tef 10000 -tmsf 20000 -ps -pss 21 -s $s"
#done

SAC_N_CONTROLLER_ABSOLUTE_PATH="/home/sschoepp/Documents/openai/controllers/sacv2/mod/sacv2_n_controller.py"

for s in {0..1}
do
  tmux new-session -d -s sacf-$s "CUDA_VISIBLE_DEVICES=0 python $SAC_N_CONTROLLER_ABSOLUTE_PATH -a -c -e FetchreachEnv-v0 -ps -pss 61 -s $s --resumable -t 2000000 -tef 10000 -tmsf 2000"
done

for s in {2..3}
do
  tmux new-session -d -s sacf-$s "CUDA_VISIBLE_DEVICES=1 python $SAC_N_CONTROLLER_ABSOLUTE_PATH -a -c -e FetchreachEnv-v0 -ps -pss 61 -s $s --resumable -t 2000000 -tef 10000 -tmsf 2000"
done

#for s in {5..9}
#do
#  tmux new-session -d -s sac-$s "CUDA_VISIBLE_DEVICES=1 python $SAC_N_CONTROLLER_ABSOLUTE_PATH -a -c -ps -pss 61 -s $s --resumable"
#done