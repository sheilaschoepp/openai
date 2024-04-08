SAC_CONTROLLER_ABSOLUTE_PATH="/home/sschoepp/Documents/openai/controllers/sac/sac_n_controller.py"

#!/bin/bash

# Number of GPUs
num_gpus=6

# Number of tmux processes per GPU
num_tmux_per_gpu=4

# Total number of tmux processes
total_tmux=$((num_gpus * num_tmux_per_gpu))

# Loop through each GPU
for ((gpu=0; gpu<num_gpus; gpu++)); do
    gpu_id=$((gpu + 1))  # GPU IDs start from 1
    echo "Launching tmux processes for GPU $gpu_id"

    # Loop to launch tmux processes
    for ((i=0; i<num_tmux_per_gpu; i++)); do
        tmux new-session -d -s "gpu${gpu_id}_process$i" "CUDA_VISIBLE_DEVICES=$gpu_id python $SAC_CONTROLLER_ABSOLUTE_PATH -e FetchReachEnv-v0 -t 2000000 -tef 10000 -tmsf 10000 -a -c -ps -pss 61"
    done
done

echo "All tmux processes launched."