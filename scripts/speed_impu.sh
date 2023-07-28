python -u main.py \
    --task_name imputation \
    --is_training 0 \
    --model TimesNet\
    --trained_model ''\
    \
    --root_path ./dataset/competition/train-5min/ \
    --flow_data_path flow-5min.csv \
    --speed_data_path speed-5min.csv \
    --dataloader_type speed \
    --freq t \
    --data_shrink 3 \
    \
    --seq_len 156 \
    --label_len 0 \
    --pred_len 0 \
    \
    --e_layers 2\
    --enc_in 40\
    --dec_in 40\
    --c_out 40\
    --d_model 64 \
    --d_ff 64 \
    --top_k 5 \
    --num_kernels 6 \
    --embed timeF \
    --kernel_factor 4 \
    --dropout 0.1 \
    --trans_layers 1 \
    --nheads 4 \
    --t_ff 128 \
    \
    --batch_size 32\
    --patience 40 \
    --learning_rate 0.001\
    --use_amp False\
    \
    --gpu 0 \

python -u main.py \
    --task_name imputation \
    --is_training 0 \
    --model TimesNet\
    --trained_model ''\
    \
    --root_path ./dataset/competition/train-5min/ \
    --flow_data_path flow-5min.csv \
    --speed_data_path speed-5min.csv \
    --dataloader_type speed \
    --freq t \
    --data_shrink 3 \
    \
    --seq_len 156 \
    --label_len 0 \
    --pred_len 0 \
    \
    --e_layers 2\
    --enc_in 40\
    --dec_in 40\
    --c_out 40\
    --d_model 64 \
    --d_ff 64 \
    --top_k 5 \
    --num_kernels 6 \
    --embed timeF \
    --kernel_factor 4 \
    --dropout 0.1 \
    --trans_layers 2 \
    --nheads 4 \
    --t_ff 128 \
    \
    --batch_size 32\
    --patience 40 \
    --learning_rate 0.001\
    --use_amp False\
    \
    --gpu 0 \


python -u main.py \
    --task_name imputation \
    --is_training 0 \
    --model TimesNet\
    --trained_model ''\
    \
    --root_path ./dataset/competition/train-5min/ \
    --flow_data_path flow-5min.csv \
    --speed_data_path speed-5min.csv \
    --dataloader_type speed \
    --freq t \
    --data_shrink 3 \
    \
    --seq_len 156 \
    --label_len 0 \
    --pred_len 0 \
    \
    --e_layers 2\
    --enc_in 40\
    --dec_in 40\
    --c_out 40\
    --d_model 128 \
    --d_ff 128 \
    --top_k 5 \
    --num_kernels 6 \
    --embed timeF \
    --kernel_factor 4 \
    --dropout 0.1 \
    --trans_layers 1 \
    --nheads 4 \
    --t_ff 128 \
    \
    --batch_size 16\
    --patience 40 \
    --learning_rate 0.0008\
    --use_amp False\
    \
    --gpu 0 \