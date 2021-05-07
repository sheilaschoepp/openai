#!/bin/bash

PPO_CONTROLLER_ABSOLUTE_PATH="/home/sschoepp/Documents/openai/controllers/ppov2/ppov2_n_controller.py"

for s in {0..29}
do
  label=$((s))
  tmux new-session -d -s ppo$label "python $PPO_CONTROLLER_ABSOLUTE_PATH -lrd --resumable -s $s -t 800000 -ef 200000"
done

#for s in {0..20}
#do
#  label=$((s+30))
#  tmux new-session -d -s ppo$label "python $PPO_CONTROLLER_ABSOLUTE_PATH --resumable -s $s"
#done

#SAC_CONTROLLER_ABSOLUTE_PATH="/home/sschoepp/Documents/openai/controllers/sacv1/sacv1_n_controller.py"
#
#for s in 0 1
#do
#  tmux new-session -d -s sac$s "CUDA_VISIBLE_DEVICES=0 python $SAC_CONTROLLER_ABSOLUTE_PATH -c --resumable -s $s"
#done