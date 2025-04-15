#!/bin/bash

SCRIPT="/home/sschoepp/Documents/openai/controllers/sac/sac_n_controller.py"
ARGS="--n_env_name=Ant-v5 \
      --n_time_steps=3000000 \
      --gamma=0.9815 \
      --tau=0.05151 \
      --alpha=0.07461 \
      --lr=0.0002225 \
      --hidden_dim=256 \
      --replay_buffer_size=1000000 \
      --batch_size=512 \
      --target_update_interval=8 \
      --time_step_eval_frequency=30000 \
      --wandb"

for SEED in {0..9}; do
  CPU=$((SEED))  # simple one-to-one mapping (seed0→CPU0, seed1→CPU1, etc.)

  echo "Starting tmux session: sac_seed${SEED} on CPU ${CPU}"
  tmux new-session -d -s "sac_seed${SEED}" \
    "taskset -c ${CPU} python ${SCRIPT} ${ARGS} --seed ${SEED}"

  sleep 30
done
