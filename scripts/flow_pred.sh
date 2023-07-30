export NUMEXPR_MAX_THREADS=128
torchrun --standalone --nproc_per_node=6 main.py \
    --task_name prediction \
    --is_training 0 \
    --model TimesNet\
    --trained_model ''\
    \
    --root_path ./dataset/competition/train-5min/ \
    --flow_data_path flow-5min.csv \
    --speed_data_path speed-5min.csv \
    --dataloader_type flow \
    --freq t \
    --data_shrink 1 \
    \
    --seq_len 36 \
    --label_len 0 \
    --pred_len 12 \
    \
    --e_layers 2\
    --enc_in 40\
    --dec_in 40\
    --c_out 40\
    --d_model 256 \
    --d_ff 256 \
    --top_k 5 \
    --num_kernels 6 \
    --embed timeF \
    --kernel_factor 2 \
    --dropout 0.1 \
    --trans_layers 1 \
    --nheads 4 \
    --t_ff 128 \
    \
    --batch_size 32\
    --patience 30 \
    --learning_rate 0.0008\
    --use_amp False\
    \
    --devices 0,1,2,3,4,5 \
    --use_multi_gpu True \


export NUMEXPR_MAX_THREADS=128
torchrun --standalone --nproc_per_node=6 main.py \
    --task_name prediction \
    --is_training 0 \
    --model TimesNet\
    --trained_model ''\
    \
    --root_path ./dataset/competition/train-5min/ \
    --flow_data_path flow-5min.csv \
    --speed_data_path speed-5min.csv \
    --dataloader_type speed \
    --freq t \
    --data_shrink 1 \
    \
    --seq_len 36 \
    --label_len 0 \
    --pred_len 12 \
    \
    --e_layers 2\
    --enc_in 40\
    --dec_in 40\
    --c_out 40\
    --d_model 256 \
    --d_ff 256 \
    --top_k 5 \
    --num_kernels 6 \
    --embed timeF \
    --kernel_factor 2 \
    --dropout 0.1 \
    --trans_layers 1 \
    --nheads 4 \
    --t_ff 128 \
    \
    --batch_size 32\
    --patience 30 \
    --learning_rate 0.0008\
    --use_amp False\
    \
    --devices 0,1,2,3,4,5 \
    --use_multi_gpu True \