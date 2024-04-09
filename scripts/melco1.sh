#!/bin/bash

# Constants
SAC_AB_CONTROLLER_ABSOLUTE_PATH="/home/sschoepp/Documents/openai/controllers/sac/sac_ab_controller.py"
FOLDER_PATH="/home/sschoepp/Documents/openai/data/SACv2_FetchReachEnv-v0:2000000_g:0.8097_t:0.0721_a:0.2_lr:0.001738_hd:256_rbs:10000_bs:512_mups:1_tui:1_tef:10000_ee:10_tmsf:20000_a:True_d:cuda_ps:True_pss:21"
ENVIRONMENTS=("FetchReachEnv-v4" "FetchReachEnv-v6")
COMMANDS=("-crb -rn" "-crb" "-rn" "")
GPUS=(0 1 2 3 4 5)
MAX_SESSIONS_PER_GPU=4
MAX_TOTAL_SESSIONS=24

# Function to check and wait for available session slots
function check_sessions {
    while true; do
        total_sessions=$(tmux ls 2>/dev/null | wc -l)
        if [ "$total_sessions" -lt "$MAX_TOTAL_SESSIONS" ]; then
            break
        fi
        echo "Maximum sessions reached. Waiting..."
        sleep 10
    done
}

# Function to find an available GPU
function find_available_gpu {
    for gpu in "${GPUS[@]}"; do
        gpu_sessions=$(tmux ls 2>/dev/null | grep -c "gpu${gpu}_")
        if [ "$gpu_sessions" -lt "$MAX_SESSIONS_PER_GPU" ]; then
            echo $gpu
            return
        fi
    done
    echo "-1"
}

# Main execution loop
for env in "${ENVIRONMENTS[@]}"; do
    for seed in {0..29}; do
        for cmd in "${COMMANDS[@]}"; do
            check_sessions
            available_gpu=$(find_available_gpu)
            if [ "$available_gpu" -eq "-1" ]; then
                echo "No GPUs available currently. Waiting..."
                while [ $(find_available_gpu) -eq "-1" ]; do
                    sleep 10
                done
                available_gpu=$(find_available_gpu)
            fi

            # Prepare session details
            session_name="gpu${available_gpu}_env_${env}_seed_${seed}_cmd_${cmd// /_}"
            log_file="$FOLDER_PATH/logs/${session_name}.log"
            full_command="CUDA_VISIBLE_DEVICES=$available_gpu python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -e $env -t 300000 $cmd -c -f $FOLDER_PATH/seed${seed} -d 2>&1 | tee $log_file"

            # Launch tmux session
            tmux new-session -d -s "$session_name" "$full_command"
            echo "Launched $session_name on GPU $available_gpu."
        done
    done
    echo "Completed all seeds for environment $env."
done

echo "All sessions scheduled."
