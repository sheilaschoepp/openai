#!/bin/bash

SCRIPT="/home/sschoepp/Documents/openai/controllers/ppo/ppo_n_controller.py"
ARGS="--n_env_name=FetchReachDense-v3 \
      --n_time_steps=40000 \
      --lr=0.0006056 \
      --linear_lr_decay \
      --gamma=0.8113 \
      --num_samples=1024 \
      --mini_batch_size=32 \
      --epochs=10 \
      --epsilon=0.2908 \
      --vf_loss_coef=0.8985 \
      --policy_entropy_coef=0.05265 \
      --max_grad_norm=0.5 \
      --gae_lambda=0.0 \
      --normalize_rewards \
      --wandb"
# set gae_lambda to 0.0 since use_gae is false

for SEED in {0..29}; do
  CPU=$((SEED))  # simple one-to-one mapping (seed0→CPU0, seed1→CPU1, etc.)

  echo "Starting tmux session: seed${SEED} on CPU ${CPU}"
  tmux new-session -d -s "seed${SEED}" \
    "taskset -c ${CPU} python ${SCRIPT} ${ARGS} --seed ${SEED}"

  sleep 30
done
