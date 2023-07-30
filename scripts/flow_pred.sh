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
    --data_shrink 2 \
    \
    --seq_len 36 \
    --label_len 0 \
    --pred_len 12 \
    \
    --e_layers 2\
    --enc_in 40\
    --dec_in 40\
    --c_out 40\
    --d_model 512 \
    --d_ff 512 \
    --top_k 5 \
    --num_kernels 6 \
    --embed timeF \
    --dropout 0.1 \
    \
    --batch_size 32\
    --patience 30 \
    --learning_rate 0.0001\
    --use_amp 0\
    \
    --devices 0,1 \
    --use_multi_gpu True \