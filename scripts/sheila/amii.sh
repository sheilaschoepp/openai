SAC_AB_CONTROLLER_ABSOLUTE_PATH="/home/sschoepp/Documents/openai/controllers/sacv2/mod/sacv2_ab_controller.py"

FILE="/media/sschoepp/easystore/shared/ant/seeds/faulty/sac/v3/new/SACv2_AntEnv-v3:20000000_Ant-v2:20000000_g:0.9646_t:0.0877_a:0.2_lr:0.001092_hd:256_rbs:500000_bs:512_mups:1_tui:1_tef:100000_ee:10_tmsf:100000_crb:True_rn:True_a:True_d:cuda_r_mod"
tmux new-session -d -s sac "python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -c --resume -rf $FILE/seed17; python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -c --resume -rf $FILE/seed18; python $SAC_AB_CONTROLLER_ABSOLUTE_PATH -c --resume -rf $FILE/seed19"
