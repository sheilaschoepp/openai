#!/bin/bash

# Define your scripts (if you have more than one).
SCRIPTS=(
  "/home/sschoepp/Documents/openai/controllers/ppo/ppo_n_controller.py"
)

# Define corresponding argument sets (each item matches a script above).
ARGS_LIST=(
  "--ab_env_name=FetchReach-F1 --ab_time_steps=50000 --file=/home/sschoepp/Documents/openai/data/PPO_FetchReach-F0:50000_lr:0.0008641_lrd:True_g:0.8301_ns:256_mbs:32_epo:10_eps:0.2887_c1:0.141_c2:0.0138_cvf:False_mgn:0.5_gae:True_lam:0.9039_nr:True_hd:64_lstd:0.0_tef:500_ee:10_d:cpu_wb --wandb"
  "--ab_env_name=FetchReach-F1 --ab_time_steps=50000 --file=/home/sschoepp/Documents/openai/data/PPO_FetchReach-F0:50000_lr:0.0008641_lrd:True_g:0.8301_ns:256_mbs:32_epo:10_eps:0.2887_c1:0.141_c2:0.0138_cvf:False_mgn:0.5_gae:True_lam:0.9039_nr:True_hd:64_lstd:0.0_tef:500_ee:10_d:cpu_wb --wandb --clear_memory"
  "--ab_env_name=FetchReach-F1 --ab_time_steps=50000 --file=/home/sschoepp/Documents/openai/data/PPO_FetchReach-F0:50000_lr:0.0008641_lrd:True_g:0.8301_ns:256_mbs:32_epo:10_eps:0.2887_c1:0.141_c2:0.0138_cvf:False_mgn:0.5_gae:True_lam:0.9039_nr:True_hd:64_lstd:0.0_tef:500_ee:10_d:cpu_wb --wandb --reinitialize_networks"
  "--ab_env_name=FetchReach-F1 --ab_time_steps=50000 --file=/home/sschoepp/Documents/openai/data/PPO_FetchReach-F0:50000_lr:0.0008641_lrd:True_g:0.8301_ns:256_mbs:32_epo:10_eps:0.2887_c1:0.141_c2:0.0138_cvf:False_mgn:0.5_gae:True_lam:0.9039_nr:True_hd:64_lstd:0.0_tef:500_ee:10_d:cpu_wb --wandb --clear_memory --reinitialize_networks"
  "--ab_env_name=FetchReach-F2 --ab_time_steps=50000 --file=/home/sschoepp/Documents/openai/data/PPO_FetchReach-F0:50000_lr:0.0008641_lrd:True_g:0.8301_ns:256_mbs:32_epo:10_eps:0.2887_c1:0.141_c2:0.0138_cvf:False_mgn:0.5_gae:True_lam:0.9039_nr:True_hd:64_lstd:0.0_tef:500_ee:10_d:cpu_wb --wandb"
  "--ab_env_name=FetchReach-F2 --ab_time_steps=50000 --file=/home/sschoepp/Documents/openai/data/PPO_FetchReach-F0:50000_lr:0.0008641_lrd:True_g:0.8301_ns:256_mbs:32_epo:10_eps:0.2887_c1:0.141_c2:0.0138_cvf:False_mgn:0.5_gae:True_lam:0.9039_nr:True_hd:64_lstd:0.0_tef:500_ee:10_d:cpu_wb --wandb --clear_memory"
  "--ab_env_name=FetchReach-F2 --ab_time_steps=50000 --file=/home/sschoepp/Documents/openai/data/PPO_FetchReach-F0:50000_lr:0.0008641_lrd:True_g:0.8301_ns:256_mbs:32_epo:10_eps:0.2887_c1:0.141_c2:0.0138_cvf:False_mgn:0.5_gae:True_lam:0.9039_nr:True_hd:64_lstd:0.0_tef:500_ee:10_d:cpu_wb --wandb --reinitialize_networks"
  "--ab_env_name=FetchReach-F2 --ab_time_steps=50000 --file=/home/sschoepp/Documents/openai/data/PPO_FetchReach-F0:50000_lr:0.0008641_lrd:True_g:0.8301_ns:256_mbs:32_epo:10_eps:0.2887_c1:0.141_c2:0.0138_cvf:False_mgn:0.5_gae:True_lam:0.9039_nr:True_hd:64_lstd:0.0_tef:500_ee:10_d:cpu_wb --wandb --clear_memory --reinitialize_networks"
)

# Number of seeds to run for each script/arg set
NUM_SEEDS=30

# Maximum tmux sessions to allow
MAX_SESSIONS=20

# Iterate over all script–argument pairs
for i in "${!SCRIPTS[@]}"; do
  SCRIPT="${SCRIPTS[$i]}"
  ARGS="${ARGS_LIST[$i]}"

  # Iterate over seed values
  for SEED in $(seq 0 $((NUM_SEEDS - 1))); do

    # Wait until fewer than MAX_SESSIONS tmux sessions are active
    while true; do
      SESSION_COUNT=$(tmux ls 2>/dev/null | wc -l) || true
      # If tmux has no sessions, 'tmux ls' errors out, so we redirect to /dev/null.

      if [ "$SESSION_COUNT" -lt "$MAX_SESSIONS" ]; then
        break
      else
        sleep 60
      fi
    done

    # Simple one-to-one mapping of seed to CPU core
    CPU=$((SEED))

    SESSION_NAME="seed${SEED}_set${i}"
    echo "Starting tmux session: ${SESSION_NAME} on CPU ${CPU}"

    tmux new-session -d -s "${SESSION_NAME}" \
      "taskset -c ${CPU} python ${SCRIPT} ${ARGS} --seed ${SEED}"

    # Short sleep to stagger startup
    sleep 30
  done
done
