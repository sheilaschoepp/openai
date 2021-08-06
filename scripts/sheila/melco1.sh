SAC_N_CONTROLLER_ABSOLUTE_PATH="/home/sschoepp/Documents/openai/controllers/sacv2/mod/sacv2_n_controller.py"

for s in 14 15 16 17
do
  tmux new-session -d -s sacv2-$s "CUDA_VISIBLE_DEVICES=6 python $SAC_N_CONTROLLER_ABSOLUTE_PATH -e FetchReachEnv-v0 -a -c -t 2000000 -tef 10000 -tmsf 20000 -ps -pss 21"
done

for s in 18 19 20 21
do
  tmux new-session -d -s sacv2-$s "CUDA_VISIBLE_DEVICES=7 python $SAC_N_CONTROLLER_ABSOLUTE_PATH -e FetchReachEnv-v0 -a -c -t 2000000 -tef 10000 -tmsf 20000 -ps -pss 21"
done