#!/bin/bash

SCRIPT="/home/sschoepp/Documents/openai/controllers/sac/sac_n_controller.py"
ARGS="--n_env_name=FetchReach-F0 \
      --n_time_steps=10000 \
      --gamma=0.8504 \
      --tau=0.003237 \
      --alpha=0.1336 \
      --lr=0.0008507 \
      --hidden_dim=256 \
      --replay_buffer_size=100000 \
      --batch_size=256 \
      --normalize_rewards \
      --target_update_interval=5 \
      --time_step_eval_frequency=100 \
      --wandb"

for SEED in {0..29}; do
  CPU=$((SEED))  # simple one-to-one mapping (seed0→CPU0, seed1→CPU1, etc.)

  echo "Starting tmux session: seed${SEED} on CPU ${CPU}"
  tmux new-session -d -s "seed${SEED}" \
    "taskset -c ${CPU} python ${SCRIPT} ${ARGS} --seed ${SEED}"

  sleep 30
done
