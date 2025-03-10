#!/bin/bash

SCRIPT="/home/sschoepp/Documents/openai/controllers/ppo/ppo_n_controller.py"
ARGS="--ab_env_name=FetchReach-F1 \
      --ab_time_steps=50000 \
      --file=/home/sschoepp/Documents/openai/data/PPO_FetchReach-F0:50000_lr:0.0008641_lrd:True_g:0.8301_ns:256_mbs:32_epo:10_eps:0.2887_c1:0.141_c2:0.0138_cvf:False_mgn:0.5_gae:True_lam:0.9039_nr:True_hd:64_lstd:0.0_tef:500_ee:10_d:cpu_wb \
      --wandb"

for SEED in {0..29}; do
  CPU=$((SEED))  # simple one-to-one mapping (seed0→CPU0, seed1→CPU1, etc.)

  echo "Starting tmux session: seed${SEED} on CPU ${CPU}"
  tmux new-session -d -s "seed${SEED}" \
    "taskset -c ${CPU} python ${SCRIPT} ${ARGS} --seed ${SEED}"

  sleep 30
done
