#!/bin/bash

PPO_N_CONTROLLER_ABSOLUTE_PATH="/home/sschoepp/Documents/openai/controllers/ppov2/ppov2_n_controller.py"

for s in 0
do
  tmux new-session -d -s ppo$s "python $PPO_N_CONTROLLER_ABSOLUTE_PATH -e FetchPickAndPlaceEnv-v0 -lrd --resumable -s $s -t 100000000"
done

for s in 1
do
  tmux new-session -d -s ppo$s "python $PPO_N_CONTROLLER_ABSOLUTE_PATH -e FetchPickAndPlaceEnv-v1 -lrd --resumable -s $s -t 100000000"
done

for s in 2
do
  tmux new-session -d -s ppo$s "python $PPO_N_CONTROLLER_ABSOLUTE_PATH -e FetchPickAndPlaceEnv-v2 -lrd --resumable -s $s -t 100000000"
done

for s in 3
do
  tmux new-session -d -s ppo$s "python $PPO_N_CONTROLLER_ABSOLUTE_PATH -e FetchReach-v1 -lrd --resumable -s $s -t 100000000 --lr 0.0003 --policy_entropy_coef 0.0 -ns 2000 -mbs 50 --epsilon 0.2; test.py"
done

SAC_CONTROLLER_ABSOLUTE_PATH="/home/sschoepp/Documents/openai/controllers/sacv2/sacv2_n_controller.py"

for s in 0
do
  tmux new-session -d -s sac$s "CUDA_VISIBLE_DEVICES=1 python $SAC_CONTROLLER_ABSOLUTE_PATH -a -c -e FetchPickAndPlaceEnv-v0 --resumable -s $s -t 5000000"
done

for s in 1
do
  tmux new-session -d -s sac$s "CUDA_VISIBLE_DEVICES=1 python $SAC_CONTROLLER_ABSOLUTE_PATH -a -c -e FetchPickAndPlaceEnv-v1 --resumable -s $s -t 5000000"
done

for s in 2
do
  tmux new-session -d -s sac$s "CUDA_VISIBLE_DEVICES=1 python $SAC_CONTROLLER_ABSOLUTE_PATH -a -c -e FetchPickAndPlaceEnv-v2 --resumable -s $s -t 5000000"
done

for s in 3
do
  tmux new-session -d -s sac$s "CUDA_VISIBLE_DEVICES=1 python $SAC_CONTROLLER_ABSOLUTE_PATH -a -c -e FetchReach-v1 -ef 10000 --resumable -s $s -t 5000000; python test.py"
done