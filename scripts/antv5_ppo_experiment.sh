#!/bin/bash

SCRIPT="/home/sschoepp/Documents/openai/controllers/ppo/ppo_n_controller.py"
ARGS="--n_env_name=Ant-v5 \
      --n_time_steps=40000000 \
      --lr=0.0001672 \
      --linear_lr_decay \
      --gamma=0.9960 \
      --num_samples=4096 \
      --mini_batch_size=32 \
      --epochs=5 \
      --epsilon=0.2458 \
      --vf_loss_coef=0.4853 \
      --policy_entropy_coef=0.003953 \
      --max_grad_norm=0.5 \
      --use_gae \
      --gae_lambda=0.9006 \
      --normalize_rewards \
      --time_step_eval_frequency=400000 \
      --wandb"

for SEED in {20..29}; do
  CPU=$((SEED))  # simple one-to-one mapping (seed0→CPU0, seed1→CPU1, etc.)

  echo "Starting tmux session: ppo_seed${SEED} on CPU ${CPU}"
  tmux new-session -d -s "ppo_seed${SEED}" \
    "taskset -c ${CPU} python ${SCRIPT} ${ARGS} --seed ${SEED}"

  sleep 30
done
