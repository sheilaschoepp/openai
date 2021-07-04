echo 'SAC_N_CONTROLLER_ABSOLUTE_PATH="/home/sschoepp/Documents/openai/controllers/sacv2/sacv2_n_controller.py"' > melco2.sh

#for i in 99
#do
#  echo 'tmux new-session -d -s sac'$i'a "CUDA_VISIBLE_DEVICES=0 python $SAC_N_CONTROLLER_ABSOLUTE_PATH -a -c -e FetchReach-v1 -s 0 -t 2000000 -tef 10000 -ps -pss '$i'; CUDA_VISIBLE_DEVICES=0 python $SAC_N_CONTROLLER_ABSOLUTE_PATH -a -c -e FetchReach-v1 -s 1 -t 2000000 -tef 10000 -ps -pss '$i'; CUDA_VISIBLE_DEVICES=0 python $SAC_N_CONTROLLER_ABSOLUTE_PATH -a -c -e FetchReach-v1 -s 2 -t 2000000 -tef 10000 -ps -pss '$i'; CUDA_VISIBLE_DEVICES=0 python $SAC_N_CONTROLLER_ABSOLUTE_PATH -a -c -e FetchReach-v1 -s 3 -t 2000000 -tef 10000 -ps -pss '$i'; CUDA_VISIBLE_DEVICES=0 python $SAC_N_CONTROLLER_ABSOLUTE_PATH -a -c -e FetchReach-v1 -s 4 -t 2000000 -tef 10000 -ps -pss '$i'"' >> melco2.sh
#  echo 'tmux new-session -d -s sac'$i'b "CUDA_VISIBLE_DEVICES=7 python $SAC_N_CONTROLLER_ABSOLUTE_PATH -a -c -e FetchReach-v1 -s 5 -t 2000000 -tef 10000 -ps -pss '$i'; CUDA_VISIBLE_DEVICES=7 python $SAC_N_CONTROLLER_ABSOLUTE_PATH -a -c -e FetchReach-v1 -s 6 -t 2000000 -tef 10000 -ps -pss '$i'; CUDA_VISIBLE_DEVICES=7 python $SAC_N_CONTROLLER_ABSOLUTE_PATH -a -c -e FetchReach-v1 -s 7 -t 2000000 -tef 10000 -ps -pss '$i'; CUDA_VISIBLE_DEVICES=7 python $SAC_N_CONTROLLER_ABSOLUTE_PATH -a -c -e FetchReach-v1 -s 8 -t 2000000 -tef 10000 -ps -pss '$i'; CUDA_VISIBLE_DEVICES=7 python $SAC_N_CONTROLLER_ABSOLUTE_PATH -a -c -e FetchReach-v1 -s 9 -t 2000000 -tef 10000 -ps -pss '$i'"' >> melco2.sh
#done

for i in 45 46 47 48
do
  echo 'tmux new-session -d -s sac'$i' "CUDA_VISIBLE_DEVICES=1 python $SAC_N_CONTROLLER_ABSOLUTE_PATH -a -c -e FetchReach-v1 -s 9 -t 2000000 -tef 10000 -ps -pss '$i'"' >> melco2.sh
done

#echo 'PPO_N_CONTROLLER_ABSOLUTE_PATH="/home/sschoepp/Documents/openai/controllers/ppov2/ppov2_n_controller.py"' > melco2.sh
#
#for i in {0..29}
#do
#  echo 'tmux new-session -d -s ppod'$i' "python $PPO_N_CONTROLLER_ABSOLUTE_PATH -lrd -s '$i'"' >> melco2.sh
#done