#!/bin/bash

#PPO_N_CONTROLLER_ABSOLUTE_PATH="/home/sschoepp/Documents/openai/controllers/ppov2/ppov2_n_controller.py"
#
#for ps in 8 9 10 11 12 13 14
#do
#  for s in {0..4}
#  do
#    tmux new-session -d -s ppo$ps$s "python $PPO_N_CONTROLLER_ABSOLUTE_PATH -lrd -s $s -t 100000000 -ps -pss $ps"
#  done
#done
#
#SAC_CONTROLLER_ABSOLUTE_PATH="/home/sschoepp/Documents/openai/controllers/sacv2/sacv2_n_controller.py"
#
#for ps in 8
#do
#  for s in {0..4}
#  do
#    tmux new-session -d -s sac$ps$s "CUDA_VISIBLE_DEVICES=0 python $SAC_CONTROLLER_ABSOLUTE_PATH -a -c --resumable -s $s -t 5000000 -ps -pss $ps"
#  done
#done
#
#for ps in 9
#do
#  for s in {0..4}
#  do
#    tmux new-session -d -s sac$ps$s "CUDA_VISIBLE_DEVICES=1 python $SAC_CONTROLLER_ABSOLUTE_PATH -a -c --resumable -s $s -t 5000000 -ps -pss $ps"
#  done
#done

PPO_N_CONTROLLER_ABSOLUTE_PATH="/home/sschoepp/Documents/openai/controllers/ppov2/ppov2_n_controller.py"

for ps in {8..12}
do
  for s in {0..4}
  do
    tmux new-session -d -s ppo$ps$s "python $PPO_N_CONTROLLER_ABSOLUTE_PATH -lrd -s $s -t 100000000 -ps -pss $ps"
  done
done

SAC_CONTROLLER_ABSOLUTE_PATH="/home/sschoepp/Documents/openai/controllers/sacv2/sacv2_n_controller.py"

for ps in 21
do
  for s in {0..4}
  do
    tmux new-session -d -s sac$ps$s "CUDA_VISIBLE_DEVICES=0 python $SAC_CONTROLLER_ABSOLUTE_PATH -a -c --resumable -s $s -t 5000000 -ps -pss $ps"
  done
done

for ps in 22
do
  for s in {0..4}
  do
    tmux new-session -d -s sac$ps$s "CUDA_VISIBLE_DEVICES=1 python $SAC_CONTROLLER_ABSOLUTE_PATH -a -c --resumable -s $s -t 5000000 -ps -pss $ps"
  done
done