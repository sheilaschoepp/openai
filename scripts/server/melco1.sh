SAC_N_CONTROLLER_ABSOLUTE_PATH="/home/sschoepp/Documents/openai/controllers/sacv2/sacv2_n_controller.py"

tmux new-session -d -s sac22 "CUDA_VISIBLE_DEVICES=6 python $SAC_N_CONTROLLER_ABSOLUTE_PATH -a -c -e FetchReach-v1 -s 9 -t 2000000 -tef 10000 -ps -pss 22"
