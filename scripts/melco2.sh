#!/bin/bash

PPO_N_CONTROLLER_ABSOLUTE_PATH="/home/sschoepp/Documents/openai/controllers/ppov2/ppov2_n_controller.py"

for ps in {0..9}
do
for s in {0..4}
do
  label=$((s))
  tmux new-session -d -s ppo$label "python $PPO_N_CONTROLLER_ABSOLUTE_PATH -lrd --resumable -s $s -t 800000 -ef 200000"
done