#!/bin/bash

# Constants
SAC_AB_CONTROLLER_ABSOLUTE_PATH="/home/sschoepp/Documents/openai/controllers/sac/sac_ab_controller.py"
FOLDER_PATH="/home/sschoepp/Documents/openai/data/SACv2_FetchReachEnv-v0:2000000_g:0.8097_t:0.0721_a:0.2_lr:0.001738_hd:256_rbs:10000_bs:512_mups:1_tui:1_tef:10000_ee:10_tmsf:20000_a:True_d:cuda_ps:True_pss:21"
ENVIRONMENTS=("FetchReachEnv-v4" "FetchReachEnv-v6")
COMMANDS=("-crb -rn" "-crb" "-rn" "")
GPUS=(0 1 2 3 4 5)
MAX_SESSIONS_PER_GPU=4

# Function to check and maintain tmux session limits
check_and_wait_for_session_availability() {
    while : ; do
        current_sessions=$(tmux ls 2>/dev/null | wc -l)
        if [ "$current_sessions" -lt 24 ]; then
            break
        fi
        echo "Waiting for available tmux session slots..."
        sleep 60
    done
}

# Launch tmux sessions for the given environment and seed
launch_sessions() {
    local env=$1
    local seed=$2
    local gpu=$3

    # Iterate over each command modification
    for cmd in "${COMMANDS[@]}"; do
        # Check for total tmux sessions cap
        check_and_wait_for_session_availability

        # Define session specifics
        session_name="gpu${gpu}_env_${env}_seed_${seed}_cmd_${cmd// /_}"
        folder_path="${FOLDER_PATH}/seed${seed}"
        full_command="CUDA_VISIBLE_DEVICES=$gpu python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -e $env -t 300000 $cmd -c -f $folder_path -d"

        # Start tmux session
        tmux new-session -d -s "$session_name" "$full_command"
        echo "Launched tmux session: $session_name on GPU $gpu with command $cmd."
    done
}

# Main execution loop
for env in "${ENVIRONMENTS[@]}"; do
    for seed in {0..29}; do
        for gpu in "${GPUS[@]}"; do
            current_gpu_sessions=$(tmux ls | grep "gpu${gpu}_" | wc -l)
            if [ "$current_gpu_sessions" -lt "$MAX_SESSIONS_PER_GPU" ]; then
                launch_sessions "$env" "$seed" "$gpu"
            fi
        done
    done
done

echo "All scheduled sessions are launched."
