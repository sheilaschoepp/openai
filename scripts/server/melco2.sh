#!/bin/bash

PPO_N_CONTROLLER_ABSOLUTE_PATH="/home/sschoepp/Documents/openai/controllers/ppov2/ppov2_n_controller.py"

for s in {0..4}
do
  tmux new-session -d -s ppo$s "python $PPO_N_CONTROLLER_ABSOLUTE_PATH -e FetchPickAndPlaceEnv-v0 -lrd --resumable -s $s -t 100000000"
done

SAC_CONTROLLER_ABSOLUTE_PATH="/home/sschoepp/Documents/openai/controllers/sacv2/sacv2_n_controller.py"

for s in {0..4}
do
  tmux new-session -d -s sac$s "CUDA_VISIBLE_DEVICES=1 python $SAC_CONTROLLER_ABSOLUTE_PATH -a -c -e FetchPickAndPlaceEnv-v0 --resumable -s $s -t 5000000"
done