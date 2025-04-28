#!/bin/bash

# Script(s) to run
SCRIPTS=(
  "/home/sschoepp/Documents/openai/controllers/sac/sac_ab_controller.py"
)

# Argument sets (everything except the --file=... portion)
BASE_ARGS_LIST=(
# melco1
#  "--ab_env_name=Ant-F1 --ab_time_steps=12000000 --wandb"
#  "--ab_env_name=Ant-F1 --ab_time_steps=12000000 --wandb --clear_replay_buffer"
#  "--ab_env_name=Ant-F1 --ab_time_steps=12000000 --wandb --reinitialize_networks"
#  "--ab_env_name=Ant-F1 --ab_time_steps=12000000 --wandb --reinitialize_networks --clear_replay_buffer"
#  "--ab_env_name=Ant-F2 --ab_time_steps=12000000 --wandb"
#  "--ab_env_name=Ant-F2 --ab_time_steps=12000000 --wandb --clear_replay_buffer"
# melco2
#  "--ab_env_name=Ant-F2 --ab_time_steps=12000000 --wandb --reinitialize_networks"
#  "--ab_env_name=Ant-F2 --ab_time_steps=12000000 --wandb --reinitialize_networks --clear_replay_buffer"
#  "--ab_env_name=Ant-F3 --ab_time_steps=12000000 --wandb"
#  "--ab_env_name=Ant-F3 --ab_time_steps=12000000 --wandb --clear_replay_buffer"
#  "--ab_env_name=Ant-F3 --ab_time_steps=12000000 --wandb --reinitialize_networks"
#  "--ab_env_name=Ant-F3 --ab_time_steps=12000000 --wandb --reinitialize_networks --clear_replay_buffer"
# ur3
#  "--ab_env_name=Ant-F4 --ab_time_steps=12000000 --wandb"
#  "--ab_env_name=Ant-F4 --ab_time_steps=12000000 --wandb --clear_replay_buffer"
#  "--ab_env_name=Ant-F4 --ab_time_steps=12000000 --wandb --reinitialize_networks"
# amii
  "--ab_env_name=Ant-F4 --ab_time_steps=12000000 --wandb --reinitialize_networks --clear_replay_buffer"
)

# Base file path (without the /seed part)
FILE_BASE="/home/sschoepp/Documents/openai/data/SAC_Ant-v5:3000000_g:0.9815_t:0.05151_a:0.07461_lr:0.0002225_hd:256_rbs:1000000_bs:512_nr:False_mups:1_tui:8_a:False_tef:30000_ee:10_d:cpu_wb"

# Number of seeds to run for each script/arg set
NUM_SEEDS=5
# Maximum tmux sessions allowed
MAX_SESSIONS=6

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
