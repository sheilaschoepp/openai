#!/bin/bash

SCRIPT="/home/sschoepp/Documents/openai/controllers/ppo/ppo_n_controller.py"
ARGS="--n_env_name=FetchReach-F0 \
      --n_time_steps=25000 \
      --lr=0.0008641 \
      --linear_lr_decay \
      --gamma=0.8301 \
      --num_samples=256 \
      --mini_batch_size=32 \
      --epochs=10 \
      --epsilon=0.2887 \
      --vf_loss_coef=0.1410 \
      --policy_entropy_coef=0.01380 \
      --max_grad_norm=0.5 \
      --use_gae \
      --gae_lambda=0.9039 \
      --normalize_rewards \
      --time_step_eval_frequency=500 \
      --wandb"

for SEED in {0..29}; do
  CPU=$((SEED))  # simple one-to-one mapping (seed0→CPU0, seed1→CPU1, etc.)

  echo "Starting tmux session: seed${SEED} on CPU ${CPU}"
  tmux new-session -d -s "seed${SEED}" \
    "taskset -c ${CPU} python ${SCRIPT} ${ARGS} --seed ${SEED}"

  sleep 30
done
