PPO_N_CONTROLLER_ABSOLUTE_PATH="/home/sschoepp/Documents/openai/controllers/ppov2/ppov2_n_controller.py"

for s in {10..29}
do
  tmux new-session -d -s sac-$s "python $PPO_N_CONTROLLER_ABSOLUTE_PATH -e FetchReachEnv-v0 -lrd -t 6000000 -tef 30000 -tmsf 60000  -ps -pss 43"
done