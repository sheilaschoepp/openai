#!/bin/bash

SAC_CONTROLLER_ABSOLUTE_PATH="/home/sschoepp/Documents/openai/controllers/sacv2/sacv2_n_controller.py"

for ps in 10
do
  for s in {0..4}
  do
    tmux new-session -d -s sac$ps$s "CUDA_VISIBLE_DEVICES=0 python $SAC_CONTROLLER_ABSOLUTE_PATH -a -c -s $s -t 5000000 -ps -pss $ps"
  done
done

for ps in 11
do
  for s in {0..4}
  do
    tmux new-session -d -s sac$ps$s "CUDA_VISIBLE_DEVICES=1 python $SAC_CONTROLLER_ABSOLUTE_PATH -a -c -s $s -t 5000000  -ps -pss $ps"
  done
done

for ps in 12
do
  for s in {0..4}
  do
    tmux new-session -d -s sac$ps$s "CUDA_VISIBLE_DEVICES=3 python $SAC_CONTROLLER_ABSOLUTE_PATH -a -c -s $s -t 5000000  -ps -pss $ps"
  done
done

for ps in 13
do
  for s in {0..4}
  do
    tmux new-session -d -s sac$ps$s "CUDA_VISIBLE_DEVICES=4 python $SAC_CONTROLLER_ABSOLUTE_PATH -a -c -s $s -t 5000000  -ps -pss $ps"
  done
done

for ps in 14
do
  for s in {0..4}
  do
    tmux new-session -d -s sac$ps$s "CUDA_VISIBLE_DEVICES=6 python $SAC_CONTROLLER_ABSOLUTE_PATH -a -c -s $s -t 5000000  -ps -pss $ps"
  done
done

for ps in 15
do
  for s in {0..4}
  do
    tmux new-session -d -s sac$ps$s "CUDA_VISIBLE_DEVICES=7 python $SAC_CONTROLLER_ABSOLUTE_PATH -a -c -s $s -t 5000000  -ps -pss $ps"
  done
done

#for ps in 2
#do
#  for s in {0..4}
#  do
#    tmux new-session -d -s sac$ps$s "CUDA_VISIBLE_DEVICES=0 python $SAC_CONTROLLER_ABSOLUTE_PATH -a -c -s $s -t 5000000 -ps -pss $ps"
#  done
#done
#
#for ps in 3
#do
#  for s in {0..4}
#  do
#    tmux new-session -d -s sac$ps$s "CUDA_VISIBLE_DEVICES=1 python $SAC_CONTROLLER_ABSOLUTE_PATH -a -c -s $s -t 5000000  -ps -pss $ps"
#  done
#done
#
#for ps in 4
#do
#  for s in {0..4}
#  do
#    tmux new-session -d -s sac$ps$s "CUDA_VISIBLE_DEVICES=3 python $SAC_CONTROLLER_ABSOLUTE_PATH -a -c -s $s -t 5000000  -ps -pss $ps"
#  done
#done
#
#for ps in 5
#do
#  for s in {0..4}
#  do
#    tmux new-session -d -s sac$ps$s "CUDA_VISIBLE_DEVICES=4 python $SAC_CONTROLLER_ABSOLUTE_PATH -a -c -s $s -t 5000000  -ps -pss $ps"
#  done
#done
#
#for ps in 6
#do
#  for s in {0..4}
#  do
#    tmux new-session -d -s sac$ps$s "CUDA_VISIBLE_DEVICES=6 python $SAC_CONTROLLER_ABSOLUTE_PATH -a -c -s $s -t 5000000  -ps -pss $ps"
#  done
#done
#
#for ps in 7
#do
#  for s in {0..4}
#  do
#    tmux new-session -d -s sac$ps$s "CUDA_VISIBLE_DEVICES=7 python $SAC_CONTROLLER_ABSOLUTE_PATH -a -c -s $s -t 5000000  -ps -pss $ps"
#  done
#done