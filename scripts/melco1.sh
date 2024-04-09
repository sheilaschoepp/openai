#!/bin/bash

# Constants
SAC_AB_CONTROLLER_ABSOLUTE_PATH="/home/sschoepp/Documents/openai/controllers/sac/sac_ab_controller.py"
FOLDER_PATH="/home/sschoepp/Documents/openai/data/SACv2_FetchReachEnv-v0:2000000_g:0.8097_t:0.0721_a:0.2_lr:0.001738_hd:256_rbs:10000_bs:512_mups:1_tui:1_tef:10000_ee:10_tmsf:20000_a:True_d:cuda_ps:True_pss:21"
ENVIRONMENTS=("FetchReachEnv-v4" "FetchReachEnv-v6")
COMMANDS=("-crb -rn" "-crb" "-rn" "")
GPUS=(0 1 2 3 4 5)
MAX_SESSIONS_PER_GPU=4
MAX_TOTAL_SESSIONS=24

# Function to launch tmux sessions
launch_sessions() {
    local env=$1
    local seed=$2
    local gpu=$3
    local cmd=$4
    local session_name="gpu${gpu}_env_${env}_seed_${seed}_cmd_${cmd// /_}"
    local folder_path="${FOLDER_PATH}/seed${seed}"
    local full_command="CUDA_VISIBLE_DEVICES=$gpu python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -e $env -t 300000 $cmd -c -f $folder_path -d 2>&1"

    # Start tmux session
    tmux new-session -d -s "$session_name" "$full_command"
    echo "Launched: $session_name"
}

# Main execution loop
for env in "${ENVIRONMENTS[@]}"; do
    for seed in {0..29}; do
        for gpu in "${GPUS[@]}"; do
            for cmd in "${COMMANDS[@]}"; do
                current_sessions=$(tmux ls 2>/dev/null | wc -l)
                if [ "$current_sessions" -ge "$MAX_TOTAL_SESSIONS" ]; then
                    echo "Reached max total sessions limit. Waiting..."
                    while [ $(tmux ls 2>/dev/null | wc -l) -ge "$MAX_TOTAL_SESSIONS" ]; do
                        sleep 10
                    done
                fi

                current_gpu_sessions=$(tmux ls | grep -c "gpu${gpu}_")
                if [ "$current_gpu_sessions" -lt "$MAX_SESSIONS_PER_GPU" ]; then
                    launch_sessions "$env" "$seed" "$gpu" "$cmd"
                else
                    echo "GPU $gpu full with $current_gpu_sessions sessions."
                fi
            done
        done
        echo "Completed all commands for seed $seed in environment $env."
    done
    echo "Completed all seeds for environment $env."
done

echo "All scheduled sessions are launched."
