#!/bin/bash

# Constants
PPO_AB_CONTROLLER_ABSOLUTE_PATH="/home/sschoepp/Documents/openai/controllers/ppo/ppo_ab_controller.py"
FOLDER_PATH="/home/sschoepp/Documents/openai/data/PPOv2_FetchReachEnv-v0:6000000_lr:0.000275_lrd:True_g:0.848_ns:3424_mbs:8_epo:24_eps:0.3_c1:1.0_c2:0.0007_cvl:False_mgn:0.5_gae:True_lam:0.9327_hd:64_lstd:0.0_tef:30000_ee:10_tmsf:60000_d:cpu_ps:True_pss:43"
ENVIRONMENTS=("FetchReachEnv-v4" "FetchReachEnv-v6")
COMMANDS=("-cm -rn" "-cm" "-rn" "")
#LOG_DIR="/home/sschoepp/logs"  # Directory to store log files

# Ensure the log directory exists
#mkdir -p $LOG_DIR

# Function to launch tmux sessions
launch_session() {
    local env=$1
    local seed=$2
    local cmd_option=$3
    local session_name="env_${env}_seed_${seed}_cmd_${cmd_option// /_}"
    local folder="${FOLDER_PATH}/seed${seed}"
#    local log_file="${LOG_DIR}/${session_name}.log"
    local full_command="python $PPO_AB_CONTROLLER_ABSOLUTE_PATH -e $env -t 6000000 $cmd_option -f $folder -d"

    tmux new-session -d -s "$session_name" "$full_command"
    echo "Launched tmux session: $session_name."
}

# Main loop to create tmux sessions
for env in "${ENVIRONMENTS[@]}"; do
    for seed in {0..29}; do
        for cmd in "${COMMANDS[@]}"; do
            # Check active tmux sessions and wait if they reach the limit
            while [ $(tmux ls | wc -l) -ge 50 ]; do
                echo "Maximum number of tmux sessions reached. Waiting..."
                sleep 60
            done
            launch_session "$env" "$seed" "$cmd"
        done
    done
done

echo "All sessions are scheduled."
