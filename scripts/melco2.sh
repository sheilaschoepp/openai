#!/bin/bash

PPO_N_CONTROLLER_ABSOLUTE_PATH="/home/sschoepp/Documents/openai/controllers/ppov2/ppov2_n_controller.py"

for ps in 0 1 2 3 4 5 6 7
do
  for s in {0..4}
  do
    label=$((s))
    tmux new-session -d -s ppo$label "python $PPO_N_CONTROLLER_ABSOLUTE_PATH -lrd -s $s -t 100000000"
  done
done

SAC_CONTROLLER_ABSOLUTE_PATH="/home/sschoepp/Documents/openai/controllers/sacv2/sacv2_n_controller.py"

for s in {15..19}
do
  tmux new-session -d -s sac$s "CUDA_VISIBLE_DEVICES=0 python $SAC_CONTROLLER_ABSOLUTE_PATH -a -c -s $s -t 5000000"
done

for s in {20..24}
do
  tmux new-session -d -s sac$s "CUDA_VISIBLE_DEVICES=1 python $SAC_CONTROLLER_ABSOLUTE_PATH -a -c -s $s -t 5000000"
done
