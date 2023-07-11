python -u main.py \
    --is_training 0 \
    --root_path ./dataset/competition/train-5min/ \
    --flow_data_path flow-5min.csv \
    --speed_data_path speed-5min.csv \
    --d_model 64 \
    --d_ff 32 \
    --gpu 0 \
    --num_kernels 6 \
    --top_k 5 \
    --lradj type3 \
    --patience 50 \
    --freq h \
    --learning_rate 0.001\
    --batch_size 16\
    --e_layers 2\
    --dropout 0.2\
    --lradj_factor 10\

python -u main.py \
    --is_training 0 \
    --root_path ./dataset/competition/train-5min/ \
    --flow_data_path flow-5min.csv \
    --speed_data_path speed-5min.csv \
    --d_model 64 \
    --d_ff 64 \
    --gpu 0 \
    --num_kernels 6 \
    --top_k 3 \
    --lradj type3 \
    --patience 50 \
    --freq h \
    --learning_rate 0.001\
    --batch_size 16\
    --e_layers 2\
    --dropout 0.2\
    --lradj_factor 10\    

python -u main.py \
    --is_training 0 \
    --root_path ./dataset/competition/train-5min/ \
    --flow_data_path flow-5min.csv \
    --speed_data_path speed-5min.csv \
    --d_model 128 \
    --d_ff 64 \
    --gpu 0 \
    --num_kernels 6 \
    --top_k 3 \
    --lradj type3 \
    --patience 50 \
    --freq h \
    --learning_rate 0.0008\
    --batch_size 16\
    --e_layers 2\
    --dropout 0.2\
    --lradj_factor 10\    


python -u main.py \
    --is_training 0 \
    --root_path ./dataset/competition/train-5min/ \
    --flow_data_path flow-5min.csv \
    --speed_data_path speed-5min.csv \
    --d_model 64 \
    --d_ff 32 \
    --gpu 0 \
    --num_kernels 6 \
    --top_k 3 \
    --lradj type3 \
    --patience 50 \
    --freq h \
    --learning_rate 0.001\
    --batch_size 16\
    --e_layers 2\
    --dropout 0.2\
    --lradj_factor 10\