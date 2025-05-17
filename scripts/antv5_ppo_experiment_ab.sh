#!/bin/bash

# Script(s) to run
SCRIPTS=(
  "/home/sschoepp/Documents/openai/controllers/ppo/ppo_ab_controller.py"
)

# Argument sets (everything except the --file=... portion)
BASE_ARGS_LIST=(
#  melco2
  "--ab_env_name=Ant-F1 --ab_time_steps=400000000 --wandb"
  "--ab_env_name=Ant-F1 --ab_time_steps=400000000 --wandb --clear_memory"
  "--ab_env_name=Ant-F1 --ab_time_steps=400000000 --wandb --reinitialize_networks"
  "--ab_env_name=Ant-F1 --ab_time_steps=400000000 --wandb --reinitialize_networks --clear_memory"
  "--ab_env_name=Ant-F2 --ab_time_steps=400000000 --wandb"
  "--ab_env_name=Ant-F2 --ab_time_steps=400000000 --wandb --clear_memory"
# amii
#  "--ab_env_name=Ant-F2 --ab_time_steps=400000000 --wandb --reinitialize_networks"
#  "--ab_env_name=Ant-F2 --ab_time_steps=400000000 --wandb --reinitialize_networks --clear_memory"
# ur3
#  "--ab_env_name=Ant-F3 --ab_time_steps=400000000 --wandb"
#  "--ab_env_name=Ant-F3 --ab_time_steps=400000000 --wandb --clear_memory"
#  "--ab_env_name=Ant-F3 --ab_time_steps=400000000 --wandb --reinitialize_networks"
#  "--ab_env_name=Ant-F3 --ab_time_steps=400000000 --wandb --reinitialize_networks --clear_memory"
#  melco1
#  "--ab_env_name=Ant-F4 --ab_time_steps=400000000 --wandb"
#  "--ab_env_name=Ant-F4 --ab_time_steps=400000000 --wandb --clear_memory"
#  "--ab_env_name=Ant-F4 --ab_time_steps=400000000 --wandb --reinitialize_networks"
#  "--ab_env_name=Ant-F4 --ab_time_steps=400000000 --wandb --reinitialize_networks --clear_memory"
)

# Base file path (without the /seed part)
FILE_BASE="/home/sschoepp/Documents/openai/data/PPO_Ant-v5:20000000_lr:0.0001672_lrd:True_g:0.996_ns:4096_mbs:32_epo:5_eps:0.2458_c1:0.4853_c2:0.003953_cvf:False_mgn:0.5_gae:True_lam:0.9006_nr:True_hd:64_lstd:0.0_tef:200000_ee:10_d:cpu_wb"

# Number of seeds to run for each script/arg set
NUM_SEEDS=5
# Maximum tmux sessions allowed
MAX_SESSIONS=31

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

      SESSION_NAME="seed${SEED}_argset${j}"
      # Append "/seed${SEED}" to the file path
      FILE_PATH="${FILE_BASE}/seed${SEED}"

      echo "Starting tmux session: ${SESSION_NAME}"
      tmux new-session -d -s "${SESSION_NAME}" \
        "python ${SCRIPT} ${BASE_ARGS} --file=${FILE_PATH}"

      # Pause 30 seconds between each new Python call
      sleep 30
    done
  done
done
