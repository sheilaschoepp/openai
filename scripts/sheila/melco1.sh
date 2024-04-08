#!/bin/bash

# Paths and environment setup
SAC_CONTROLLER_ABSOLUTE_PATH="/home/sschoepp/Documents/openai/controllers/sac/sac_n_controller.py"
LOG_DIR="/home/sschoepp/logs"

# Make sure the log directory exists
mkdir -p $LOG_DIR

# Number of GPUs and maximum tmux sessions per GPU at any given time
num_gpus=6
max_tmux_per_gpu=4

# Total number of tmux processes to be launched
total_tmux=30

# Initialize GPU usage count
declare -A gpu_usage
for ((gpu=0; gpu<num_gpus; gpu++)); do
    gpu_usage[$gpu]=0
done

# Launching tmux sessions
session_count=0
while [ $session_count -lt $total_tmux ]; do
    for ((gpu=0; gpu<num_gpus && session_count<total_tmux; gpu++)); do
        if [ ${gpu_usage[$gpu]} -lt $max_tmux_per_gpu ]; then
            session_name="gpu${gpu}_process${gpu_usage[$gpu]}"
            log_file="${LOG_DIR}/${session_name}.log"
            tmux new-session -d -s "$session_name" "CUDA_VISIBLE_DEVICES=$gpu python $SAC_CONTROLLER_ABSOLUTE_PATH -e FetchReachEnv-v0 -t 2000000 -tef 10000 -tmsf 10000 -a -c -ps -pss 61 -d 2>&1 | tee $log_file"
            echo "Session $session_name started, logging to $log_file."
            gpu_usage[$gpu]=$((gpu_usage[$gpu]+1))
            session_count=$((session_count+1))
        fi
    done
done

echo "All tmux processes launched."
