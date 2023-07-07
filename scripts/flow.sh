python -u main.py \
    --is_training 0 \
    --root_path ./dataset/competition/train-5min/ \
    --data_path flow.csv \
    --d_model 32 \
    --d_ff 32 \
    --gpu 1 \
    --num_kernels 3 \
    --top_k 3 \
    --lradj type3 \
    --batch_size 32 \
    --patience 10 \
    --freq h \

python -u main.py \
    --is_training 0 \
    --root_path ./dataset/competition/train-5min/ \
    --data_path flow.csv \
    --d_model 64 \
    --d_ff 64 \
    --gpu 1 \
    --num_kernels 6 \
    --top_k 5 \
    --lradj type3 \
    --batch_size 32 \
    --patience 10 \
    --freq h \

python -u main.py \
    --is_training 0 \
    --root_path ./dataset/competition/train-5min/ \
    --data_path flow.csv \
    --d_model 64 \
    --d_ff 64 \
    --gpu 1 \
    --num_kernels 6 \
    --top_k 3 \
    --lradj type3 \
    --batch_size 32 \
    --patience 10 \
    --freq t \