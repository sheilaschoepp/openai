SAC_AB_CONTROLLER_ABSOLUTE_PATH="/home/taghianj/Documents/openai/controllers/sacv2/mod/sacv2_ab_controller.py"
RESUME_FILE="/home/taghianj/Documents/data_openai/ant/normal/SACv2_Ant-v2:20000000_g:0.9646_t:0.0877_a:0.2_lr:0.001092_hd:256_rbs:500000_bs:512_mups:1_tui:1_tef:100000_ee:10_tmsf:200000_a:True_d:cuda_ps:True_pss:61_resumed_mod"

# ------------------------ V1

#tmux new-session -d -s sacv1-0 "CUDA_VISIBLE_DEVICES=0 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -c -e AntEnv-v1 -t 20000000 --resumable -f $RESUME_FILE/seed0 -d"
#tmux new-session -d -s sacv1-1 "CUDA_VISIBLE_DEVICES=0 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -c -e AntEnv-v1 -t 20000000 --resumable -f $RESUME_FILE/seed1 -d"
#tmux new-session -d -s sacv1-2 "CUDA_VISIBLE_DEVICES=0 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -c -e AntEnv-v1 -t 20000000 --resumable -f $RESUME_FILE/seed2 -d"
#tmux new-session -d -s sacv1-3 "CUDA_VISIBLE_DEVICES=0 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -c -e AntEnv-v1 -t 20000000 --resumable -f $RESUME_FILE/seed3 -d"
#tmux new-session -d -s sacv1-4 "CUDA_VISIBLE_DEVICES=0 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -c -e AntEnv-v1 -t 20000000 --resumable -f $RESUME_FILE/seed4 -d"
#tmux new-session -d -s sacv1-5 "CUDA_VISIBLE_DEVICES=0 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -c -e AntEnv-v1 -t 20000000 --resumable -f $RESUME_FILE/seed5 -d"
#tmux new-session -d -s sacv1-6 "CUDA_VISIBLE_DEVICES=0 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -c -e AntEnv-v1 -t 20000000 --resumable -f $RESUME_FILE/seed6 -d"
#tmux new-session -d -s sacv1-7 "CUDA_VISIBLE_DEVICES=1 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -c -e AntEnv-v1 -t 20000000 --resumable -f $RESUME_FILE/seed7 -d"
#tmux new-session -d -s sacv1-8 "CUDA_VISIBLE_DEVICES=1 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -c -e AntEnv-v1 -t 20000000 --resumable -f $RESUME_FILE/seed8 -d"
#tmux new-session -d -s sacv1-9 "CUDA_VISIBLE_DEVICES=1 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -c -e AntEnv-v1 -t 20000000 --resumable -f $RESUME_FILE/seed9 -d"

# TOP PRIORITY

#tmux new-session -d -s sacv1-10 "CUDA_VISIBLE_DEVICES=6 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -c -e AntEnv-v1 -f $RESUME_FILE/seed10 -t 20000000 --resumable -d"
#tmux new-session -d -s sacv1-11 "CUDA_VISIBLE_DEVICES=6 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -c -e AntEnv-v1 -f $RESUME_FILE/seed11 -t 20000000 --resumable -d"
#tmux new-session -d -s sacv1-12 "CUDA_VISIBLE_DEVICES=6 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -c -e AntEnv-v1 -f $RESUME_FILE/seed12 -t 20000000 --resumable -d"
tmux new-session -d -s sacv1-13 "CUDA_VISIBLE_DEVICES=6 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -c -e AntEnv-v1 -f $RESUME_FILE/seed13 -t 20000000 --resumable -d"
tmux new-session -d -s sacv1-14 "CUDA_VISIBLE_DEVICES=6 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -c -e AntEnv-v1 -f $RESUME_FILE/seed14 -t 20000000 --resumable -d"
tmux new-session -d -s sacv1-10_crb "CUDA_VISIBLE_DEVICES=6 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -c -e AntEnv-v1 -f $RESUME_FILE/seed10 -t 20000000 --resumable -crb -d"
tmux new-session -d -s sacv1-11_crb "CUDA_VISIBLE_DEVICES=6 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -c -e AntEnv-v1 -f $RESUME_FILE/seed11 -t 20000000 --resumable -crb -d"
tmux new-session -d -s sacv1-12_crb "CUDA_VISIBLE_DEVICES=6 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -c -e AntEnv-v1 -f $RESUME_FILE/seed12 -t 20000000 --resumable -crb -d"
tmux new-session -d -s sacv1-13_crb "CUDA_VISIBLE_DEVICES=6 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -c -e AntEnv-v1 -f $RESUME_FILE/seed13 -t 20000000 --resumable -crb -d"
tmux new-session -d -s sacv1-14_crb "CUDA_VISIBLE_DEVICES=6 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -c -e AntEnv-v1 -f $RESUME_FILE/seed14 -t 20000000 --resumable -crb -d"
tmux new-session -d -s sacv1-10_rn "CUDA_VISIBLE_DEVICES=7 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -c -e AntEnv-v1 -f $RESUME_FILE/seed10 -t 20000000 --resumable -rn -d"
tmux new-session -d -s sacv1-11_rn "CUDA_VISIBLE_DEVICES=7 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -c -e AntEnv-v1 -f $RESUME_FILE/seed11 -t 20000000 --resumable -rn -d"
tmux new-session -d -s sacv1-12_rn "CUDA_VISIBLE_DEVICES=7 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -c -e AntEnv-v1 -f $RESUME_FILE/seed12 -t 20000000 --resumable -rn -d"
tmux new-session -d -s sacv1-13_rn "CUDA_VISIBLE_DEVICES=7 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -c -e AntEnv-v1 -f $RESUME_FILE/seed13 -t 20000000 --resumable -rn -d"
tmux new-session -d -s sacv1-14_rn "CUDA_VISIBLE_DEVICES=7 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -c -e AntEnv-v1 -f $RESUME_FILE/seed14 -t 20000000 --resumable -rn -d"
tmux new-session -d -s sacv1-10_crb_rn "CUDA_VISIBLE_DEVICES=7 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -c -e AntEnv-v1 -f $RESUME_FILE/seed10 -t 20000000 --resumable -crb -rn -d"
tmux new-session -d -s sacv1-11_crb_rn "CUDA_VISIBLE_DEVICES=7 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -c -e AntEnv-v1 -f $RESUME_FILE/seed11 -t 20000000 --resumable -crb -rn -d"
#tmux new-session -d -s sacv1-12_crb_rn "CUDA_VISIBLE_DEVICES=7 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -c -e AntEnv-v1 -f $RESUME_FILE/seed12 -t 20000000 --resumable -crb -rn -d"
#tmux new-session -d -s sacv1-13_crb_rn "CUDA_VISIBLE_DEVICES=7 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -c -e AntEnv-v1 -f $RESUME_FILE/seed13 -t 20000000 --resumable -crb -rn -d"
#tmux new-session -d -s sacv1-14_crb_rn "CUDA_VISIBLE_DEVICES=7 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -c -e AntEnv-v1 -f $RESUME_FILE/seed14 -t 20000000 --resumable -crb -rn -d"


# ------------------------ V2

#tmux new-session -d -s sacv2-0 "CUDA_VISIBLE_DEVICES=1 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -c -e AntEnv-v2 -t 20000000 --resumable -f $RESUME_FILE/seed0 -d"
#tmux new-session -d -s sacv2-1 "CUDA_VISIBLE_DEVICES=1 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -c -e AntEnv-v2 -t 20000000 --resumable -f $RESUME_FILE/seed1 -d"
#tmux new-session -d -s sacv2-2 "CUDA_VISIBLE_DEVICES=1 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -c -e AntEnv-v2 -t 20000000 --resumable -f $RESUME_FILE/seed2 -d"
#tmux new-session -d -s sacv2-3 "CUDA_VISIBLE_DEVICES=1 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -c -e AntEnv-v2 -t 20000000 --resumable -f $RESUME_FILE/seed3 -d"
#tmux new-session -d -s sacv2-4 "CUDA_VISIBLE_DEVICES=3 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -c -e AntEnv-v2 -t 20000000 --resumable -f $RESUME_FILE/seed4 -d"
#tmux new-session -d -s sacv2-5 "CUDA_VISIBLE_DEVICES=3 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -c -e AntEnv-v2 -t 20000000 --resumable -f $RESUME_FILE/seed5 -d"
#tmux new-session -d -s sacv2-6 "CUDA_VISIBLE_DEVICES=3 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -c -e AntEnv-v2 -t 20000000 --resumable -f $RESUME_FILE/seed6 -d"
#tmux new-session -d -s sacv2-7 "CUDA_VISIBLE_DEVICES=3 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -c -e AntEnv-v2 -t 20000000 --resumable -f $RESUME_FILE/seed7 -d"
#tmux new-session -d -s sacv2-8 "CUDA_VISIBLE_DEVICES=3 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -c -e AntEnv-v2 -t 20000000 --resumable -f $RESUME_FILE/seed8 -d"
#tmux new-session -d -s sacv2-9 "CUDA_VISIBLE_DEVICES=3 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -c -e AntEnv-v2 -t 20000000 --resumable -f $RESUME_FILE/seed9 -d"

# ------------------------ V3

#tmux new-session -d -s sacv3-0 "CUDA_VISIBLE_DEVICES=3 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -c -e AntEnv-v3 -t 20000000 --resumable -f $RESUME_FILE/seed0 -d"
#tmux new-session -d -s sacv3-1 "CUDA_VISIBLE_DEVICES=4 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -c -e AntEnv-v3 -t 20000000 --resumable -f $RESUME_FILE/seed1 -d"
#tmux new-session -d -s sacv3-2 "CUDA_VISIBLE_DEVICES=4 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -c -e AntEnv-v3 -t 20000000 --resumable -f $RESUME_FILE/seed2 -d"
#tmux new-session -d -s sacv3-3 "CUDA_VISIBLE_DEVICES=4 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -c -e AntEnv-v3 -t 20000000 --resumable -f $RESUME_FILE/seed3 -d"
#tmux new-session -d -s sacv3-4 "CUDA_VISIBLE_DEVICES=4 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -c -e AntEnv-v3 -t 20000000 --resumable -f $RESUME_FILE/seed4 -d"
#tmux new-session -d -s sacv3-5 "CUDA_VISIBLE_DEVICES=4 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -c -e AntEnv-v3 -t 20000000 --resumable -f $RESUME_FILE/seed5 -d"
#tmux new-session -d -s sacv3-6 "CUDA_VISIBLE_DEVICES=4 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -c -e AntEnv-v3 -t 20000000 --resumable -f $RESUME_FILE/seed6 -d"
#tmux new-session -d -s sacv3-7 "CUDA_VISIBLE_DEVICES=4 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -c -e AntEnv-v3 -t 20000000 --resumable -f $RESUME_FILE/seed7 -d"
#tmux new-session -d -s sacv3-8 "CUDA_VISIBLE_DEVICES=6 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -c -e AntEnv-v3 -t 20000000 --resumable -f $RESUME_FILE/seed8 -d"
#tmux new-session -d -s sacv3-9 "CUDA_VISIBLE_DEVICES=6 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -c -e AntEnv-v3 -t 20000000 --resumable -f $RESUME_FILE/seed9 -d"

# ------------------------ V4

#tmux new-session -d -s sacv4-0 "CUDA_VISIBLE_DEVICES=6 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -c -e AntEnv-v4 -t 20000000 --resumable -f $RESUME_FILE/seed0 -d"
#tmux new-session -d -s sacv4-1 "CUDA_VISIBLE_DEVICES=6 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -c -e AntEnv-v4 -t 20000000 --resumable -f $RESUME_FILE/seed1 -d"
#tmux new-session -d -s sacv4-2 "CUDA_VISIBLE_DEVICES=6 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -c -e AntEnv-v4 -t 20000000 --resumable -f $RESUME_FILE/seed2 -d"
#tmux new-session -d -s sacv4-3 "CUDA_VISIBLE_DEVICES=6 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -c -e AntEnv-v4 -t 20000000 --resumable -f $RESUME_FILE/seed3 -d"
#tmux new-session -d -s sacv4-4 "CUDA_VISIBLE_DEVICES=6 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -c -e AntEnv-v4 -t 20000000 --resumable -f $RESUME_FILE/seed4 -d"
#tmux new-session -d -s sacv4-5 "CUDA_VISIBLE_DEVICES=7 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -c -e AntEnv-v4 -t 20000000 --resumable -f $RESUME_FILE/seed5 -d"
#tmux new-session -d -s sacv4-6 "CUDA_VISIBLE_DEVICES=7 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -c -e AntEnv-v4 -t 20000000 --resumable -f $RESUME_FILE/seed6 -d"
#tmux new-session -d -s sacv4-7 "CUDA_VISIBLE_DEVICES=7 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -c -e AntEnv-v4 -t 20000000 --resumable -f $RESUME_FILE/seed7 -d"
#tmux new-session -d -s sacv4-8 "CUDA_VISIBLE_DEVICES=7 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -c -e AntEnv-v4 -t 20000000 --resumable -f $RESUME_FILE/seed8 -d"
#tmux new-session -d -s sacv4-9 "CUDA_VISIBLE_DEVICES=7 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -c -e AntEnv-v4 -t 20000000 --resumable -f $RESUME_FILE/seed9 -d"

# test

#tmux new-session -d -s sacv2_crb-0 "CUDA_VISIBLE_DEVICES=0 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -c -e AntEnv-v2 -t 20000000 --resumable -f $RESUME_FILE/seed0 -crb -d"
#tmux new-session -d -s sacv2_crb-1 "CUDA_VISIBLE_DEVICES=0 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -c -e AntEnv-v2 -t 20000000 --resumable -f $RESUME_FILE/seed1 -crb -d"
#tmux new-session -d -s sacv2_crb-2 "CUDA_VISIBLE_DEVICES=0 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -c -e AntEnv-v2 -t 20000000 --resumable -f $RESUME_FILE/seed2 -crb -d"
#tmux new-session -d -s sacv2_crb-3 "CUDA_VISIBLE_DEVICES=0 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -c -e AntEnv-v2 -t 20000000 --resumable -f $RESUME_FILE/seed3 -crb -d"
#tmux new-session -d -s sacv2_crb-4 "CUDA_VISIBLE_DEVICES=0 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -c -e AntEnv-v2 -t 20000000 --resumable -f $RESUME_FILE/seed4 -crb -d"
#tmux new-session -d -s sacv2_crb-5 "CUDA_VISIBLE_DEVICES=0 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -c -e AntEnv-v2 -t 20000000 --resumable -f $RESUME_FILE/seed5 -crb -d"
#tmux new-session -d -s sacv2_crb-6 "CUDA_VISIBLE_DEVICES=0 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -c -e AntEnv-v2 -t 20000000 --resumable -f $RESUME_FILE/seed6 -crb -d"
#tmux new-session -d -s sacv2_crb-7 "CUDA_VISIBLE_DEVICES=0 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -c -e AntEnv-v2 -t 20000000 --resumable -f $RESUME_FILE/seed7 -crb -d"
#tmux new-session -d -s sacv2_crb-8 "CUDA_VISIBLE_DEVICES=1 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -c -e AntEnv-v2 -t 20000000 --resumable -f $RESUME_FILE/seed8 -crb -d"
#tmux new-session -d -s sacv2_crb-9 "CUDA_VISIBLE_DEVICES=1 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -c -e AntEnv-v2 -t 20000000 --resumable -f $RESUME_FILE/seed9 -crb -d"

#tmux new-session -d -s sacv2_antv3_0 "CUDA_VISIBLE_DEVICES=3 python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -c -e AntEnv-v3 -t 20000000 --resumable -f $RESUME_FILE/seed0 -crb -rn"
