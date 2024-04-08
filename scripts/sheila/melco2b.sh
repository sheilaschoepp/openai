#!/bin/bash

# Paths and environment setup
PPO_CONTROLLER_ABSOLUTE_PATH="/home/sschoepp/Documents/openai/controllers/ppo/ppo_n_controller.py"

# Total number of tmux processes to be launched, each with a unique seed from 0 to 29
total_tmux=30

# PPOv2_FetchReachEnv-v0:6000000_lr:0.000275_lrd:True_g:0.848_ns:3424_mbs:8_epo:24_eps:0.3_c1:1.0_c2:0.0007_cvl:False_mgn:0.5_gae:True_lam:0.9327_hd:64_lstd:0.0_tef:30000_ee:10_tmsf:60000_d:cpu_ps:True_pss:43

# Launching tmux sessions
for ((seed=0; seed<total_tmux; seed++)); do
    session_name="session_seed${seed}"
    tmux new-session -d -s "$session_name" "python $PPO_CONTROLLER_ABSOLUTE_PATH -e FetchReachEnv-v0 -t 6000000 -tef 30000 -tmsf 30000 -ps -pss 43 -s $seed -d"
    echo "Session $session_name started with seed $seed."
done

echo "All tmux processes launched."
