#!/bin/bash

# Paths and environment setup
SAC_CONTROLLER_ABSOLUTE_PATH="/home/sschoepp/Documents/openai/controllers/sac/sac_n_controller.py"
#LOG_DIR="/home/sschoepp/logs"

# Make sure the log directory exists
#mkdir -p $LOG_DIR

# Define the specific GPUs to use (0-indexed as 5 and 6 for the sixth and seventh GPUs)
gpus=(5 6)  # Assuming numbering starts at 0, and these are GPU indices.
num_gpus=${#gpus[@]}  # The number of GPUs being used.

# Number of tmux sessions to be launched, one for each seed 24 to 29.
total_tmux=6

# Seed range.
start_seed=24
end_seed=29

# Initialize GPU usage count.
declare -A gpu_usage
for gpu in ${gpus[@]}; do
    gpu_usage[$gpu]=0
done

# SACv2_FetchReachEnv-v0:2000000_g:0.8097_t:0.0721_a:0.2_lr:0.001738_hd:256_rbs:10000_bs:512_mups:1_tui:1_tef:10000_ee:10_tmsf:20000_a:True_d:cuda_ps:True_pss:21
#

# Launching tmux sessions.
session_count=0
seed=$start_seed
while [ $session_count -lt $total_tmux ]; do
    for gpu in ${gpus[@]}; do
        if [ ${gpu_usage[$gpu]} -lt 3 ] && [ $seed -le $end_seed ]; then
            session_name="gpu${gpu}_seed${seed}"
#            log_file="${LOG_DIR}/${session_name}.log"
            tmux new-session -d -s "$session_name" "CUDA_VISIBLE_DEVICES=$gpu python $SAC_CONTROLLER_ABSOLUTE_PATH -e FetchReachEnv-v0 -t 2000000 -tef 10000 -tmsf 10000 -a -c -ps -pss 21 -s $seed -d"
            echo "Session $session_name started with seed $seed on GPU $gpu."
            gpu_usage[$gpu]=$((gpu_usage[$gpu]+1))
            session_count=$((session_count+1))
            seed=$((seed+1))
        fi
    done
    # Check if all seeds have been assigned.
    if [ $seed -gt $end_seed ]; then
        break
    fi
done

echo "All tmux processes launched."
