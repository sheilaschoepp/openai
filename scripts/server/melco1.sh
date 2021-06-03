#!/bin/bash

SAC_N_CONTROLLER_ABSOLUTE_PATH="/home/sschoepp/Documents/openai/controllers/sacv2/sacv2_n_controller.py"

for s in {0..4}
do
  tmux new-session -d -s sac47$s "CUDA_VISIBLE_DEVICES=0 python $SAC_N_CONTROLLER_ABSOLUTE_PATH -a -c --resumable -s $s -ps -pss 47"
done

for s in {5..9}
do
  tmux new-session -d -s sac47$s "CUDA_VISIBLE_DEVICES=1 python $SAC_N_CONTROLLER_ABSOLUTE_PATH -a -c --resumable -s $s -ps -pss 47"
done

for s in {0..4}
do
  tmux new-session -d -s sac48$s "CUDA_VISIBLE_DEVICES=3 python $SAC_N_CONTROLLER_ABSOLUTE_PATH -a -c --resumable -s $s -ps -pss 48"
done

for s in {5..9}
do
  tmux new-session -d -s sac48$s "CUDA_VISIBLE_DEVICES=4 python $SAC_N_CONTROLLER_ABSOLUTE_PATH -a -c --resumable -s $s -ps -pss 48"
done

for s in {0..4}
do
  tmux new-session -d -s sac49$s "CUDA_VISIBLE_DEVICES=6 python $SAC_N_CONTROLLER_ABSOLUTE_PATH -a -c --resumable -s $s -ps -pss 49"
done

for s in {5..9}
do
  tmux new-session -d -s sac49$s "CUDA_VISIBLE_DEVICES=7 python $SAC_N_CONTROLLER_ABSOLUTE_PATH -a -c --resumable -s $s -ps -pss 49"
done