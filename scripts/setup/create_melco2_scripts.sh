echo 'SAC_N_CONTROLLER_ABSOLUTE_PATH="/home/sschoepp/Documents/openai/controllers/sacv2/sacv2_n_controller.py"' > melco2.sh

#for i in 99
#do
#  echo 'tmux new-session -d -s sac'$i'a "CUDA_VISIBLE_DEVICES=0 python $SAC_N_CONTROLLER_ABSOLUTE_PATH -a -c -e FetchReach-v1 -s 0 -t 2000000 -tef 10000 -ps -pss '$i'; CUDA_VISIBLE_DEVICES=0 python $SAC_N_CONTROLLER_ABSOLUTE_PATH -a -c -e FetchReach-v1 -s 1 -t 2000000 -tef 10000 -ps -pss '$i'; CUDA_VISIBLE_DEVICES=0 python $SAC_N_CONTROLLER_ABSOLUTE_PATH -a -c -e FetchReach-v1 -s 2 -t 2000000 -tef 10000 -ps -pss '$i'; CUDA_VISIBLE_DEVICES=0 python $SAC_N_CONTROLLER_ABSOLUTE_PATH -a -c -e FetchReach-v1 -s 3 -t 2000000 -tef 10000 -ps -pss '$i'; CUDA_VISIBLE_DEVICES=0 python $SAC_N_CONTROLLER_ABSOLUTE_PATH -a -c -e FetchReach-v1 -s 4 -t 2000000 -tef 10000 -ps -pss '$i'"' >> melco2.sh
#  echo 'tmux new-session -d -s sac'$i'b "CUDA_VISIBLE_DEVICES=7 python $SAC_N_CONTROLLER_ABSOLUTE_PATH -a -c -e FetchReach-v1 -s 5 -t 2000000 -tef 10000 -ps -pss '$i'; CUDA_VISIBLE_DEVICES=7 python $SAC_N_CONTROLLER_ABSOLUTE_PATH -a -c -e FetchReach-v1 -s 6 -t 2000000 -tef 10000 -ps -pss '$i'; CUDA_VISIBLE_DEVICES=7 python $SAC_N_CONTROLLER_ABSOLUTE_PATH -a -c -e FetchReach-v1 -s 7 -t 2000000 -tef 10000 -ps -pss '$i'; CUDA_VISIBLE_DEVICES=7 python $SAC_N_CONTROLLER_ABSOLUTE_PATH -a -c -e FetchReach-v1 -s 8 -t 2000000 -tef 10000 -ps -pss '$i'; CUDA_VISIBLE_DEVICES=7 python $SAC_N_CONTROLLER_ABSOLUTE_PATH -a -c -e FetchReach-v1 -s 9 -t 2000000 -tef 10000 -ps -pss '$i'"' >> melco2.sh
#done

#for s in {10..14}
#do
#  echo 'tmux new-session -d -s sac'$s'a "CUDA_VISIBLE_DEVICES=3 python $SAC_N_CONTROLLER_ABSOLUTE_PATH -a -c -e FetchReach-v1 -s '$s' -t 2000000 -tef 10000 -ps -pss 21"' >> melco2.sh
#done
#for s in {15..19}
#do
#  echo 'tmux new-session -d -s sac'$s'a "CUDA_VISIBLE_DEVICES=4 python $SAC_N_CONTROLLER_ABSOLUTE_PATH -a -c -e FetchReach-v1 -s '$s' -t 2000000 -tef 10000 -ps -pss 21"' >> melco2.sh
#done
#for s in {20..24}
#do
#  echo 'tmux new-session -d -s sac'$s'a "CUDA_VISIBLE_DEVICES=6 python $SAC_N_CONTROLLER_ABSOLUTE_PATH -a -c -e FetchReach-v1 -s '$s' -t 2000000 -tef 10000 -ps -pss 21"' >> melco2.sh
#done
#for s in {25..29}
#do
#  echo 'tmux new-session -d -s sac'$s'a "CUDA_VISIBLE_DEVICES=7 python $SAC_N_CONTROLLER_ABSOLUTE_PATH -a -c -e FetchReach-v1 -s '$s' -t 2000000 -tef 10000 -ps -pss 21"' >> melco2.sh
#done

#echo 'RESUME_FILE="/home/sschoepp/Documents/openai/data/SACv2_Ant-v2:20000000_g:0.9646_t:0.0877_a:0.2_lr:0.001092_hd:256_rbs:500000_bs:512_mups:1_tui:1_tef:100000_ee:10_tmsf:1000000_a:True_d:cuda_ps:True_pss:61_r"' >> melco2.sh
#for s in {0..4}
#do
#  echo 'tmux new-session -d -s sac61'$s' "CUDA_VISIBLE_DEVICES=0 python $SAC_N_CONTROLLER_ABSOLUTE_PATH --resume --resume_file $RESUME_FILE/seed'$s' -t 25000000"' >> melco2.sh
#done
#for s in {5..9}
#do
#  echo 'tmux new-session -d -s sac61'$s' "CUDA_VISIBLE_DEVICES=1 python $SAC_N_CONTROLLER_ABSOLUTE_PATH --resume --resume_file $RESUME_FILE/seed'$s' -t 25000000"' >> melco2.sh
#done
#for s in {5..9}
#do
#  echo 'tmux new-session -d -s sac61'$s' "CUDA_VISIBLE_DEVICES=1 python $SAC_N_CONTROLLER_ABSOLUTE_PATH --resume --resume_file $RESUME_FILE/seed'$s' -t 25000000"' >> melco2.sh
#done

echo 'PPO_N_CONTROLLER_ABSOLUTE_PATH="/home/sschoepp/Documents/openai/controllers/ppov2/ppov2_n_controller.py"' > melco2.sh

for i in {10..29}
do
  echo 'tmux new-session -d -s ppo61'$i' "python $PPO_N_CONTROLLER_ABSOLUTE_PATH -lrd -s '$i' -ps -pss 61"' >> melco2.sh
done
