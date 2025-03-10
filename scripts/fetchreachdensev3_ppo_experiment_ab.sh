#!/bin/bash

# Script(s) to run
SCRIPTS=(
  "/home/sschoepp/Documents/openai/controllers/ppo/ppo_ab_controller.py"
)

# Argument sets (everything except the --file=... portion)
BASE_ARGS_LIST=(
  "--ab_env_name=FetchReach-F1 --ab_time_steps=50000 --wandb"
  "--ab_env_name=FetchReach-F1 --ab_time_steps=50000 --wandb --clear_memory"
  "--ab_env_name=FetchReach-F1 --ab_time_steps=50000 --wandb --reinitialize_networks"
  "--ab_env_name=FetchReach-F1 --ab_time_steps=50000 --wandb --clear_memory --reinitialize_networks"
  "--ab_env_name=FetchReach-F2 --ab_time_steps=50000 --wandb"
  "--ab_env_name=FetchReach-F2 --ab_time_steps=50000 --wandb --clear_memory"
  "--ab_env_name=FetchReach-F2 --ab_time_steps=50000 --wandb --reinitialize_networks"
  "--ab_env_name=FetchReach-F2 --ab_time_steps=50000 --wandb --clear_memory --reinitialize_networks"
)

# Base file path (without the /seed part)
FILE_BASE="/home/sschoepp/Documents/openai/data/PPO_FetchReach-F0:50000_lr:0.0008641_lrd:True_g:0.8301_ns:256_mbs:32_epo:10_eps:0.2887_c1:0.141_c2:0.0138_cvf:False_mgn:0.5_gae:True_lam:0.9039_nr:True_hd:64_lstd:0.0_tef:500_ee:10_d:cpu_wb"

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
