SAC_N_CONTROLLER_ABSOLUTE_PATH="/home/taghianj/Documents/openai/controllers/sacv2/mod/sacv2_n_controller.py"
RESUME_FILE="/local/melco2-1/shared/ant/seeds/normal/SACv2_Ant-v2:20000000_g:0.9646_t:0.0877_a:0.2_lr:0.001092_hd:256_rbs:500000_bs:512_mups:1_tui:1_tef:100000_ee:10_tmsf:200000_a:True_d:cuda_ps:True_pss:61_r_mod"

tmux new-session -d -s sacv2-0 "CUDA_VISIBLE_DEVICES=0 python $SAC_N_CONTROLLER_ABSOLUTE_PATH -c --resume -rf $RESUME_FILE/seed0"
tmux new-session -d -s sacv2-1 "CUDA_VISIBLE_DEVICES=0 python $SAC_N_CONTROLLER_ABSOLUTE_PATH -c --resume -rf $RESUME_FILE/seed1"
tmux new-session -d -s sacv2-2 "CUDA_VISIBLE_DEVICES=0 python $SAC_N_CONTROLLER_ABSOLUTE_PATH -c --resume -rf $RESUME_FILE/seed2"
tmux new-session -d -s sacv2-3 "CUDA_VISIBLE_DEVICES=0 python $SAC_N_CONTROLLER_ABSOLUTE_PATH -c --resume -rf $RESUME_FILE/seed3"
tmux new-session -d -s sacv2-4 "CUDA_VISIBLE_DEVICES=0 python $SAC_N_CONTROLLER_ABSOLUTE_PATH -c --resume -rf $RESUME_FILE/seed4"
tmux new-session -d -s sacv2-5 "CUDA_VISIBLE_DEVICES=1 python $SAC_N_CONTROLLER_ABSOLUTE_PATH -c --resume -rf $RESUME_FILE/seed5"
tmux new-session -d -s sacv2-6 "CUDA_VISIBLE_DEVICES=1 python $SAC_N_CONTROLLER_ABSOLUTE_PATH -c --resume -rf $RESUME_FILE/seed6"
tmux new-session -d -s sacv2-7 "CUDA_VISIBLE_DEVICES=1 python $SAC_N_CONTROLLER_ABSOLUTE_PATH -c --resume -rf $RESUME_FILE/seed7"
tmux new-session -d -s sacv2-8 "CUDA_VISIBLE_DEVICES=1 python $SAC_N_CONTROLLER_ABSOLUTE_PATH -c --resume -rf $RESUME_FILE/seed8"
tmux new-session -d -s sacv2-9 "CUDA_VISIBLE_DEVICES=1 python $SAC_N_CONTROLLER_ABSOLUTE_PATH -c --resume -rf $RESUME_FILE/seed9"


