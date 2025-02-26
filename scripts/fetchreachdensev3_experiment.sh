#!/bin/bash

SCRIPT="sac_n_controller.py"
ARGS="--n_env_name=Ant-v5 \
      --n_time_steps=20000 \
      --gamma=0.8004 \
      --tau=0.04758 \
      --alpha=0.03560 \
      --lr=0.0008758 \
      --hidden_dim=256 \
      --replay_buffer_size=500000 \
      --batch_size=128 \
      --normalize_rewards \
      --target_update_interval=1 \
      --time_step_eval_frequency=100"

for SEED in {0..29}; do
  CPU=$((SEED))  # simple one-to-one mapping (seed0→CPU0, seed1→CPU1, etc.)
  tmux new-session -d -s "seed${SEED}" \
    "taskset -c ${CPU} python ${SCRIPT} ${ARGS} --seed ${SEED}"
  sleep 30
done
