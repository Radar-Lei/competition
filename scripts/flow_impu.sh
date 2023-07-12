python -u main.py \
    --task_name imputation \
    --is_training 0 \
    --model TimesNet\
    \
    --root_path ./dataset/competition/train-5min/ \
    --flow_data_path flow-5min.csv \
    --speed_data_path speed-5min.csv \
    --dataloader_type flow \
    --freq h \
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
    --d_ff 512 \
    --top_k 5 \
    --num_kernels 6 \
    --dropout 0.2 \
    \
    --lradj type3 \
    --lradj_factor 20 \
    --batch_size 32\
    --patience 30 \
    --learning_rate 0.0001\
    \
    --gpu 0 \

python -u main.py \
    --task_name imputation \
    --is_training 0 \
    --model TimesNet\
    \
    --root_path ./dataset/competition/train-5min/ \
    --flow_data_path flow-5min.csv \
    --speed_data_path speed-5min.csv \
    --dataloader_type agg \
    --freq h \
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
    --d_model 512 \
    --d_ff 512 \
    --top_k 5 \
    --num_kernels 6 \
    --dropout 0.2 \
    \
    --lradj type3 \
    --lradj_factor 20 \
    --batch_size 32\
    --patience 30 \
    --learning_rate 0.001\
    \
    --gpu 0 \