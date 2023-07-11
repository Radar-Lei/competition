from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
from utils import time_features
import numpy as np

class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 flow_data_path='ETTh1.csv', speed_data_path='', scale=True, timeenc=0, freq='h', scaler=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val', 'pred']
        type_map = {'train': 0, 'val': 1, 'test': 2, 'pred': 3}
        self.set_type = type_map[flag]

        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.scaler = scaler

        self.root_path = root_path
        self.flow_data_path = flow_data_path
        self.speed_data_path = speed_data_path
        self.__read_data__()

    def __read_data__(self):
        if self.set_type != 3:
            self.scaler = StandardScaler()
        df_flow_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.flow_data_path))
        df_speed_raw = pd.read_csv(os.path.join(self.root_path, self.speed_data_path))
        
        df_raw = np.concatenate((df_flow_raw.values[:,1:], df_speed_raw.values[:,1:]),axis=1)
        df_raw = np.concatenate((np.expand_dims(df_flow_raw['date'].values, axis=1), df_raw),axis=1)
        df_raw = pd.DataFrame(df_raw)
        df_raw.rename(columns={0:'date'}, inplace=True)
        
        df_raw[df_raw == 0] = np.nan
        df_raw.fillna(method='ffill', inplace=True)
        df_raw.fillna(method='bfill', inplace=True)
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        # cols.remove(cols[-1])
        cols.remove('date')
        df_raw = df_raw[['date'] + cols]

        self.L_d = 288

        if self.set_type == 3:
            self.num_day = 7
        else:
            self.num_day = 90

        num_train = int(self.num_day * 0.8) * self.L_d
        num_test = int(self.num_day * 0.1) * self.L_d
        num_vali = len(df_raw) - num_train - num_test

        border1s = [0, num_train, len(df_raw) - num_test]
        border2s = [num_train, num_train + num_vali, len(df_raw)]

        if self.set_type != 3:
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]
        else:
            border1 = 0
            border2 = len(df_raw)

        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        if self.scale:
            if self.set_type != 3:
                train_data = df_data[border1s[0]:border2s[0]]
                self.scaler.fit(train_data.values)
                data = self.scaler.transform(df_data.values)
            else:
                data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        if (self.set_type == 0):
            s_begin = index * 6
            s_end = s_begin + self.seq_len
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len
        elif self.set_type == 3:
            s_begin = index * self.seq_len
            s_end = s_begin + self.seq_len
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len
        else:
            s_begin = 5*12 + index * self.L_d
            s_end = s_begin + self.seq_len
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        if self.set_type == 0:
            return int((len(self.data_x) - self.seq_len - self.pred_len) / 6 ) + 1
        elif self.set_type == 3: # pred
            return self.num_day
        else:
            return int(len(self.data_x) / self.L_d)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    
    def pred_transform(self):
        return self.scaler
    

def data_provider(args, flag, scaler=None):
    Data = Dataset_Custom
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test' or flag == 'val' or flag == 'pred':
        shuffle_flag = False
        drop_last = True
        
        batch_size = 1  # bsz=1 for evaluation
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq

    if flag == 'pred':
        root_path = args.pred_root_path
    else:
        root_path = args.root_path

    data_set = Data(
        root_path=root_path,
        flow_data_path=args.flow_data_path,
        speed_data_path=args.speed_data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        timeenc=timeenc,
        freq=freq,
        scaler=scaler,
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        drop_last=drop_last)
    return data_set, data_loader