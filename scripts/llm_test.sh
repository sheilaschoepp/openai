#!/bin/bash

# Constants
PPO_CONTROLLER_ABSOLUTE_PATH="controllers/ppo/ppo_ab_controller.py"
ENVIRONMENTS=("FetchReachEnv-v20")
TOTAL_SEEDS=15
WINDOWS=10

# Function to launch tmux windows and run seeds sequentially
launch_window() {
    local window_index=$1
    local window_name="window_${window_index}"
    
    # Calculate seed range for this window
    local start_seed=$((window_index * TOTAL_SEEDS / WINDOWS + 1))
    local end_seed=$(((window_index + 1) * TOTAL_SEEDS / WINDOWS))

    # Create a new tmux window
    tmux new-window -t mix_exp -n "$window_name"

    for env in "${ENVIRONMENTS[@]}"; do
        for seed in $(seq $start_seed $end_seed); do
            local full_command="python $PPO_CONTROLLER_ABSOLUTE_PATH -e $env -t 250000 --file /home/afraz1/Documents/openai/data/PPO_FetchReachEnv-v20:100000_lr:0.000275_lrd:False_slrd:1.0_g:0.848_ns:3424_mbs:8_epo:24_eps:0.3_c1:1.0_c2:0.0007_cvl:False_mgn:0.5_gae:True_lam:0.9327_hd:64_lstd:0.0_tef:1750_ee:10_tmsf:50000000_d:cpu_ps:True_pss:43"
            tmux send-keys -t mix_exp:"$window_name" "$full_command" C-m
            tmux send-keys -t mix_exp:"$window_name" "echo 'Completed: $env with seed $seed'" C-m
        done
    done

    echo "Launched tmux window $window_name running seeds $start_seed to $end_seed"
}

# Main loop to create tmux windows
tmux new-session -d -s mix_exp

for i in $(seq 0 $((WINDOWS - 1))); do
    launch_window "$i"
done

echo "All windows are scheduled in tmux."
tmux attach-session -t mix_exp