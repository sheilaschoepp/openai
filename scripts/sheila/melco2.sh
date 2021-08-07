#PPO_N_CONTROLLER_ABSOLUTE_PATH="/home/sschoepp/Documents/openai/controllers/ppov2/ppov2_n_controller.py"
#
#for s in {10..29}
#do
#  tmux new-session -d -s sac-$s "python $PPO_N_CONTROLLER_ABSOLUTE_PATH -e FetchReachEnv-v0 -lrd -t 6000000 -tef 30000 -tmsf 60000  -ps -pss 43"
#done

SAC_N_CONTROLLER_ABSOLUTE_PATH="/home/sschoepp/Documents/openai/controllers/sacv2/mod/sacv2_n_controller.py"

for s in 22 23
do
  tmux new-session -d -s sacv2-$s "CUDA_VISIBLE_DEVICES=0 python $SAC_N_CONTROLLER_ABSOLUTE_PATH -e FetchReachEnv-v0 -a -c -t 2000000 -tef 10000 -tmsf 20000 -ps -pss 21"
done

for s in 24 25
do
  tmux new-session -d -s sacv2-$s "CUDA_VISIBLE_DEVICES=1 python $SAC_N_CONTROLLER_ABSOLUTE_PATH -e FetchReachEnv-v0 -a -c -t 2000000 -tef 10000 -tmsf 20000 -ps -pss 21"
done