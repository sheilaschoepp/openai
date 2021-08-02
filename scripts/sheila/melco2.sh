SAC_N_CONTROLLER_ABSOLUTE_PATH="/home/sschoepp/Documents/openai/controllers/sacv2/mod/sacv2_n_controller.py"

for s in {0..4}
do
  tmux new-session -d -s sac-$s "CUDA_VISIBLE_DEVICES=0 python $SAC_N_CONTROLLER_ABSOLUTE_PATH -a -c --resumable -tmsf 200000 -s $s -ps -pss 61"
done

for s in {5..9}
do
  tmux new-session -d -s sac-$s "CUDA_VISIBLE_DEVICES=1 python $SAC_N_CONTROLLER_ABSOLUTE_PATH -a -c --resumable -tmsf 200000 -s $s -ps -pss 61"
done