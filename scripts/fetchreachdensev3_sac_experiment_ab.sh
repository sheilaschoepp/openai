#!/bin/bash

# Script(s) to run
SCRIPTS=(
  "/home/sschoepp/Documents/openai/controllers/sac/sac_ab_controller.py"
)

# Argument sets (everything except the --file=... portion)
BASE_ARGS_LIST=(
  "--ab_env_name=FetchReach-F1 --ab_time_steps=30000 --wandb"
  "--ab_env_name=FetchReach-F1 --ab_time_steps=30000 --wandb --clear_replay_buffer"
  "--ab_env_name=FetchReach-F1 --ab_time_steps=30000 --wandb --reinitialize_networks"
  "--ab_env_name=FetchReach-F1 --ab_time_steps=30000 --wandb --clear_replay_buffer --reinitialize_networks"
  "--ab_env_name=FetchReach-F2 --ab_time_steps=30000 --wandb"
  "--ab_env_name=FetchReach-F2 --ab_time_steps=30000 --wandb --clear_replay_buffer"
  "--ab_env_name=FetchReach-F2 --ab_time_steps=30000 --wandb --reinitialize_networks"
  "--ab_env_name=FetchReach-F2 --ab_time_steps=30000 --wandb --clear_replay_buffer --reinitialize_networks"
)

# Base file path (without the /seed part)
FILE_BASE="/home/sschoepp/Documents/openai/data/SAC_FetchReach-F0:10000_g:0.8504_t:0.003237_a:0.1336_lr:0.0008507_hd:256_rbs:100000_bs:256_nr:True_mups:1_tui:5_a:False_tef:100_ee:10_d:cpu_wb"

# Number of seeds to run for each script/arg set
NUM_SEEDS=30
# Maximum tmux sessions allowed
MAX_SESSIONS=20

# Iterate over all script–argument pairs
for i in "${!SCRIPTS[@]}"; do
  SCRIPT="${SCRIPTS[$i]}"

  for j in "${!BASE_ARGS_LIST[@]}"; do
    BASE_ARGS="${BASE_ARGS_LIST[$j]}"

    for SEED in $(seq 0 $((NUM_SEEDS - 1))); do
      # Enforce tmux session limit
      while true; do
        SESSION_COUNT=$(tmux ls 2>/dev/null | wc -l || echo 0)
        if [ "$SESSION_COUNT" -lt "$MAX_SESSIONS" ]; then
          break
        else
          echo "Reached max tmux sessions (${SESSION_COUNT}). Waiting..."
          sleep 60
        fi
      done

      CPU=$((SEED))
      SESSION_NAME="seed${SEED}_argset${j}"
      # Append "/seed${SEED}" to the file path
      FILE_PATH="${FILE_BASE}/seed${SEED}"

      echo "Starting tmux session: ${SESSION_NAME} on CPU ${CPU}"
      tmux new-session -d -s "${SESSION_NAME}" \
        "taskset -c ${CPU} python ${SCRIPT} \
          ${BASE_ARGS} \
          --file=${FILE_PATH}"

      # Pause 30 seconds between each new Python call
      sleep 30
    done
  done
done
