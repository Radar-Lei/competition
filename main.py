import argparse
import torch
import random
import numpy as np
from execution import Executer

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='TimesNet')

# basic config
parser.add_argument('--task_name', type=str, default='imputation',
                    help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
parser.add_argument('--is_training', type=int, default=0, help='status, options:[0:training, 1:testing, 2:pred]')

# data loader
parser.add_argument('--root_path', type=str, default='./dataset/competition/train-5min', help='root path of the data file') # competition
parser.add_argument('--data_path', type=str, default='flow.csv', help='data file') # flow.csv


parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')

# forecasting task
parser.add_argument('--seq_len', type=int, default=156, help='input sequence length')
parser.add_argument('--label_len', type=int, default=0, help='start token length')
parser.add_argument('--pred_len', type=int, default=0, help='prediction sequence length')

parser.add_argument('--mask_rate', type=float, default=0.5, help='mask ratio')

# model define
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--factor', type=int, default=3, help='attn factor') # whay is this?
parser.add_argument('--enc_in', type=int, default=40, help='encoder input size') # dim of feature/ num of nodes
parser.add_argument('--dec_in', type=int, default=40, help='decoder input size')
parser.add_argument('--c_out', type=int, default=40, help='output size')
parser.add_argument('--d_model', type=int, default=64, help='dimension of model') # 512
parser.add_argument('--d_ff', type=int, default=64, help='dimension of fcn') # FC network, 2048
parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock') # 5
parser.add_argument('--num_kernels', type=int, default=6, help='for Inception') # 6
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')


# optimization
parser.add_argument('--des', type=str, default='Exp', help='exp description')
parser.add_argument('--itr', type=int, default=1, help='experiments times') # num of experiments
parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1', help='device ids of multile gpus')

args = parser.parse_args()
args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

if args.is_training == 0:
    # train
    for ii in range(args.itr):
        setting = 'dm{}_df{}_el{}_dl{}_fc{}_{}_{}_{}'.format(
            args.d_model,
            args.d_ff,
            args.e_layers,
            args.d_layers,
            args.factor,
            args.des,
            ii,
            args.data_path)
        
        exp = Executer(args)
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)
        torch.cuda.empty_cache() 

elif args.is_training == 1:
    # test
    ii = 0
    setting = 'dm{}_df{}_el{}_nk{}_fq{}_{}_{}_{}'.format(
        args.d_model,
        args.d_ff,
        args.e_layers,
        args.num_kernels,
        args.freq,
        args.des,
        ii,
        args.data_path)

    exp = Executer(args)  # set experiments
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting, test=1)
    torch.cuda.empty_cache()