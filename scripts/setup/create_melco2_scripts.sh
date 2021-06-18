echo 'PPO_N_CONTROLLER_ABSOLUTE_PATH="/home/sschoepp/Documents/openai/controllers/ppov2/ppov2_n_controller.py"' > melco2.sh

for s in 5 6
do
  echo 'tmux new-session -d -s ps85'$s' "python $PPO_N_CONTROLLER_ABSOLUTE_PATH -e FetchReach-v1 -lrd -s '$s' -t 6000000 -tef 30000 -tmsf 100000 -ps -pss 85"' >> melco2.sh
done

for s in 7 8 9
do
  echo 'tmux new-session -d -s ps75'$s' "python $PPO_N_CONTROLLER_ABSOLUTE_PATH -e FetchReach-v1 -lrd -s '$s' -t 6000000 -tef 30000 -tmsf 100000 -ps -pss 75"' >> melco2.sh
done