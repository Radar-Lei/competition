from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
from utils import time_features
import numpy as np
import datetime

class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', 
                 size=None,
                 flow_data_path='ETTh1.csv', 
                 speed_data_path='', 
                 scale=True, 
                 timeenc=0, 
                 freq='h', 
                 scaler=None, 
                 loader_type='agg',
                 task_name = 'imputation',
                 data_shrink = 1):
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
        self.loader_type = loader_type
        self.task_name = task_name
        self.data_shrink = data_shrink

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
        
        if self.loader_type in ['multi','agg','agg_flow','agg_speed']:
            df_raw = np.concatenate((df_flow_raw.values[:,1:], df_speed_raw.values[:,1:]),axis=1)
            df_raw = np.concatenate((np.expand_dims(df_flow_raw['date'].values, axis=1), df_raw),axis=1)
            df_raw = pd.DataFrame(df_raw)
            df_raw.rename(columns={0:'date'}, inplace=True)
        elif self.loader_type == 'flow':
            df_raw = df_flow_raw
        else: # self.loader_type == 'speed':
            df_raw = df_speed_raw
        
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

        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        if self.loader_type in ['agg','agg_flow','agg_speed']:
            df_data_flow = df_data.iloc[:, :40]
            df_data_speed = df_data.iloc[:, 40:]
            L, _ = df_data_flow.shape
            result_flow_pred = np.sum(df_data_flow.values.reshape((L,10,4)), axis=2)
            result_speed_pred = np.sum((df_data_flow.values * df_data_speed.values).reshape((L,10,4)), axis=2) / result_flow_pred
            if self.loader_type == 'agg_flow':
                df_data = result_flow_pred
            elif self.loader_type == 'agg_speed':
                df_data = result_speed_pred
            else:
                df_data = np.hstack((result_flow_pred, result_speed_pred))
        else:
            df_data = df_data.values
        
        num_days_train = int(self.num_day * 0.8)
        num_days_test = int(self.num_day * 0.05)
        num_days_vali = self.num_day - num_days_train - num_days_test

        df_data = pd.DataFrame(df_data,index=pd.DatetimeIndex(df_raw['date'].values))

        if self.set_type != 3:
            # Get the start and end dates from df_data index
            start_date = df_data.index.min()
            end_date = df_data.index.max()

            # Generate a range of dates between start_date and end_date
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')

            # Randomly select dates for each dataset
            train_dates = np.random.choice(date_range, size=num_days_train, replace=False)
            train_dates = train_dates.astype('datetime64[D]')
            train_dates = train_dates.astype(datetime.datetime)

            vali_dates = np.random.choice(date_range[~np.isin(date_range, train_dates)], size=num_days_vali, replace=False)
            vali_dates = vali_dates.astype('datetime64[D]')
            vali_dates = vali_dates.astype(datetime.datetime)

            test_dates = np.random.choice(date_range[~np.isin(date_range, np.concatenate([train_dates, vali_dates]))], size=num_days_test, replace=False)
            test_dates = test_dates.astype('datetime64[D]')
            test_dates = test_dates.astype(datetime.datetime)


            # Filter df_data based on the selected dates
            df_train = df_data[pd.Series(df_data.index.date, index=df_data.index).isin(train_dates)]
            df_vali = df_data[pd.Series(df_data.index.date, index=df_data.index).isin(vali_dates)]
            df_test = df_data[pd.Series(df_data.index.date, index=df_data.index).isin(test_dates)]

        if self.scale:
            if self.set_type != 3:
                self.scaler.fit(df_train.values)
                train_data = self.scaler.transform(df_train.values)
                vali_data = self.scaler.transform(df_vali.values)
                test_data = self.scaler.transform(df_test.values)
            else:
                pred_data = self.scaler.transform(df_data.values)
        else:
            train_data = df_train.values
            vali_data = df_vali.values
            test_data = df_test.values

        if self.set_type == 0:
            df_stamp = df_train.reset_index().rename(columns={'index': 'date'})[['date']]
        elif self.set_type == 1:
            df_stamp = df_vali.reset_index().rename(columns={'index': 'date'})[['date']]
        elif self.set_type == 2:
            df_stamp = df_test.reset_index().rename(columns={'index': 'date'})[['date']]
        else:
            df_stamp = df_data.reset_index().rename(columns={'index': 'date'})[['date']]

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

        if self.set_type == 0:
            self.data_x = train_data
            self.data_y = train_data
        elif self.set_type == 1:
            self.data_x = vali_data
            self.data_y = vali_data
        elif self.set_type == 2:
            self.data_x = test_data
            self.data_y = test_data
        else:
            self.data_x = pred_data
            self.data_y = pred_data

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        if self.set_type != 3: # when not pred
            s_begin = index * self.data_shrink
            s_end = s_begin + self.seq_len
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len
        elif self.set_type == 3:
            if self.task_name == 'imputation':
                s_begin = index * self.seq_len
            else:
                s_begin = index * (self.seq_len + self.pred_len)

            s_end = s_begin + self.seq_len
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        if self.set_type != 3:
            return int((len(self.data_x) - self.seq_len - self.pred_len) / self.data_shrink ) + 1
        else: # self.set_type == 3: # pred
            return int(len(self.data_x) / (self.seq_len + self.pred_len))

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    
    def pred_transform(self):
        return self.scaler
    

def data_provider(args, flag, scaler=None):
    Data = Dataset_Custom
    timeenc = 0 if args.embed != 'timeF' else 1

    if (flag == 'pred') or (flag == 'test'):
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
        loader_type=args.dataloader_type,
        task_name = args.task_name,
        data_shrink=args.data_shrink,
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        drop_last=drop_last)
    return data_set, data_loader