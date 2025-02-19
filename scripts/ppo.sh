#!/bin/bash

# Define session range and maximum active sessions
SESSION_START=174 # start at 1
SESSION_END=400
MAX_ACTIVE_SESSIONS=21

# Define command to run in each session
COMMAND="python controllers/ppo/ppo_n_controller.py -o"

# Function to get the count of active tmux sessions
get_active_tmux_sessions() {
    tmux list-sessions 2>/dev/null | wc -l
}

# Main loop to create and manage sessions
for (( i=$SESSION_START; i<=$SESSION_END; i++ ))
do
    SESSION_NAME="r$i"

    # Wait until the number of active sessions is below the limit
    while [ $(get_active_tmux_sessions) -ge $MAX_ACTIVE_SESSIONS ]
    do
        echo "Maximum active sessions reached. Waiting..."
        sleep 300  # Check again every 300 seconds
    done

    # Create a new tmux session
    echo "Starting session $SESSION_NAME"
    tmux new-session -d -s "$SESSION_NAME" "$COMMAND"
    sleep 60  # Small delay to avoid overwhelming tmux
done

echo "All sessions started."
