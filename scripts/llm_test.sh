#!/bin/bash

# Constants
PPO_CONTROLLER_ABSOLUTE_PATH="controllers/ppo/ppo_a_controller.py"
ENVIRONMENTS=("FetchReachEnv-v11")
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
    tmux new-window -t my_experiments -n "$window_name"

    for env in "${ENVIRONMENTS[@]}"; do
        for seed in $(seq $start_seed $end_seed); do
            local full_command="python $PPO_CONTROLLER_ABSOLUTE_PATH -e $env -s $seed -d -ps -pss 43 -tef 5000 -t 1000000"
            tmux send-keys -t my_experiments:"$window_name" "$full_command" C-m
            tmux send-keys -t my_experiments:"$window_name" "echo 'Completed: $env with seed $seed'" C-m
        done
    done

    echo "Launched tmux window $window_name running seeds $start_seed to $end_seed"
}

# Main loop to create tmux windows
tmux new-session -d -s my_experiments

for i in $(seq 0 $((WINDOWS - 1))); do
    launch_window "$i"
done

echo "All windows are scheduled in tmux."
tmux attach-session -t my_experiments