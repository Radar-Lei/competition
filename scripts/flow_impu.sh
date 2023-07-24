python -u main.py \
    --task_name imputation \
    --is_training 0 \
    --model TimesNet\
    \
    --root_path ./dataset/competition/train-5min/ \
    --flow_data_path flow-5min.csv \
    --speed_data_path speed-5min.csv \
    --dataloader_type flow \
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
    --d_model 256 \
    --d_ff 256 \
    --top_k 5 \
    --num_kernels 6 \
    --embed timeF \
    --dropout 0.1 \
    \
    --diff_schedule quad \
    --diff_steps 100 \
    --diff_samples 100 \
    --beta_start 0.0001 \
    --bata_end 0.2 \
    --sampling_shrink_interval 4 \
    \
    --batch_size 32\
    --patience 30 \
    --learning_rate 0.0001\
    \
    --gpu 0 \