PPO_N_CONTROLLER_ABSOLUTE_PATH="/home/sschoepp/Documents/openai/controllers/ppov2/ppov2_n_controller.py"
tmux new-session -d -s ppo8 "python $PPO_N_CONTROLLER_ABSOLUTE_PATH -e FetchReach-v1 -lrd -t 6000000 -tef 30000 -tmsf 100000 -s 8 -ps -pss 43"
tmux new-session -d -s ppo9 "python $PPO_N_CONTROLLER_ABSOLUTE_PATH -e FetchReach-v1 -lrd -t 6000000 -tef 30000 -tmsf 100000 -s 9 -ps -pss 43"
