#!/bin/bash

# Define session range and maximum active sessions
SESSION_START=1
SESSION_END=600
MAX_ACTIVE_SESSIONS=24
CUDA_DEVICES=(0 1 2 3 4 5)  # List of available CUDA devices
MAX_PROCESSES_PER_DEVICE=4

# Define command to run in each session (with placeholder for CUDA device)
COMMAND="python controllers/sac/sac_n_controller.py -o"

# Function to get the count of active tmux sessions
get_active_tmux_sessions() {
    tmux list-sessions 2>/dev/null | wc -l
}

# Function to calculate the CUDA device based on the session number
get_cuda_device() {
    local session_num=$1
    local device_index=$(( (session_num - 1) % (${#CUDA_DEVICES[@]} * MAX_PROCESSES_PER_DEVICE) / MAX_PROCESSES_PER_DEVICE ))
    echo ${CUDA_DEVICES[$device_index]}
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

    # Determine the CUDA device for this session
    CUDA_DEVICE=$(get_cuda_device $i)

    # Create a new tmux session with the appropriate CUDA_VISIBLE_DEVICES
    echo "Starting session $SESSION_NAME on CUDA_VISIBLE_DEVICES=$CUDA_DEVICE"
    tmux new-session -d -s "$SESSION_NAME" "CUDA_VISIBLE_DEVICES=$CUDA_DEVICE $COMMAND"
    sleep 60  # Small delay to avoid overwhelming tmux
done

echo "All sessions started."
