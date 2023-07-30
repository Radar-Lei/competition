import torch
from exp_base import Exp_Basic
import os
import torch.nn as nn
from dataloader import data_provider
from torch import optim
import numpy as np
from utils import metric, visual, EarlyStopping
import time
import pandas as pd
import torch.distributed as dist
from torch.cuda.amp import GradScaler

class Exp_Imputation(Exp_Basic):
    def __init__(self, args):
        super(Exp_Imputation, self).__init__(args)

    def _build_model(self):

        if self.args.use_multi_gpu and self.args.use_gpu:
            dist.init_process_group(backend='nccl')
            self.local_rank = int(os.environ['LOCAL_RANK'])
            torch.cuda.set_device(self.local_rank)
            model = self.model_dict[self.args.model].Model(self.args).cuda()
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = nn.parallel.DistributedDataParallel(model, device_ids=[self.local_rank])
            if len(self.args.trained_model) > 1:
                print("loading model")
                path = os.path.join(self.args.checkpoints, self.args.trained_model)
                checkpoint = torch.load(os.path.join(path, 'checkpoint.pth'))
                model.load_state_dict(checkpoint['model_state_dict'])            
            self.scaler = GradScaler() # mixed precision training

        return model

    def _get_data(self, flag, scaler=None):
        data_set, data_loader = data_provider(self.args, flag, scaler)
        return data_set, data_loader

    def _select_optimizer(self, model):
        model_optim = optim.Adam(model.parameters(), lr=self.args.learning_rate, weight_decay=1e-5)
        if self.args.trained_model:
            print("loading optimizer")
            path = os.path.join(self.args.checkpoints, self.args.trained_model)
            checkpoint = torch.load(os.path.join(path, 'checkpoint.pth'))
            model_optim.load_state_dict(checkpoint['optimizer_state_dict'])
        return model_optim
    
    def _get_scheduler(self, optimizer):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=5, verbose=True)
        if self.args.trained_model:
            print("loading scheduler")
            path = os.path.join(self.args.checkpoints, self.args.trained_model)
            checkpoint = torch.load(os.path.join(path, 'checkpoint.pth'))
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return scheduler
    
    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = torch.tensor(0.).cuda()
        self.model.eval()

        _, K = vali_data[0][0].shape
        mask = self.create_fixed_mask(K)
        mask = torch.from_numpy(np.repeat(mask[np.newaxis, :, :], vali_loader.batch_size, axis=0))

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, actual_mask, _) in enumerate(vali_loader):
                batch_x = batch_x.float().cuda()
                batch_x_mark = batch_x_mark.float().cuda()

                # random mask
                B, T, N = batch_x.shape
                """
                B = batch size
                T = seq len
                N = number of features
                """
                # mask = torch.rand((B, T, N)).to(self.device)
                # mask[mask <= self.args.mask_rate] = 0  # masked
                # mask[mask > self.args.mask_rate] = 1  # remained
                mask = mask.cuda()
                actual_mask = actual_mask.cuda()
                target_mask = actual_mask - mask

                inp = batch_x.masked_fill(mask == 0, 0)

                outputs = self.model(inp, batch_x_mark, None, None, mask)

                f_dim = 0
                outputs = outputs[:, :, f_dim:]
                pred = outputs
                true = batch_x

                loss = criterion(pred[target_mask == 1], true[target_mask == 1])
                total_loss += loss

        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        _, K = train_data[0][0].shape
        mask = self.create_fixed_mask(K)
        mask = torch.from_numpy(np.repeat(mask[np.newaxis, :, :], train_loader.batch_size, axis=0)).cuda()

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)  

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer(self.model)
        criterion = self._select_criterion()

        folder_path = './train_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        scheduler = self._get_scheduler(model_optim)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = torch.tensor(0.).cuda()
            train_size = torch.tensor(len(train_loader)).float().cuda()
            vali_size = torch.tensor(len(vali_loader)).float().cuda()
            test_size = torch.tensor(len(test_loader)).float().cuda()

            train_loader.sampler.set_epoch(epoch) # prevent sampling bug
            self.model.train()
            epoch_time = time.time()
            # here actual_mask is for the entire sequence, where 0 in the original data masked as 0, otherwise 1
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, actual_mask, _) in enumerate(train_loader):
                iter_count += 1

                batch_x = batch_x.float().cuda()
                batch_x_mark = batch_x_mark.float().cuda()

                actual_mask = actual_mask.cuda()
                target_mask = actual_mask - mask
                target_mask[target_mask <0] = 0

                # random mask
                # B, T, N = batch_x.shape
                # mask = torch.rand((B, T, N)).to(self.device)
                # mask[mask <= self.args.mask_rate] = 0  # masked
                # mask[mask > self.args.mask_rate] = 1  # remained
                inp = batch_x.masked_fill(mask == 0, 0)

                outputs = self.model(inp, batch_x_mark, None, None, mask)

                f_dim = 0
                outputs = outputs[:, :, f_dim:]
                loss = criterion(outputs[target_mask == 1], batch_x[target_mask == 1])
                train_loss += loss

                if ((i + 1) % 100 == 0) and (self.local_rank == 0):
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                model_optim.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(model_optim)
                self.scaler.update()
            
            if ((epoch + 1) % 20 == 0) and (self.local_rank == 0):
                # eval
                outputs = outputs.detach().cpu().numpy()
                pred = outputs
                true = batch_x.detach().cpu().numpy()

                pred = train_data.inverse_transform(pred[0])
                true = train_data.inverse_transform(true[0])

                for j in range(true.shape[1]):
                    filled = true[:, j].copy()
                    filled = filled * (actual_mask-target_mask)[0, :, j].detach().cpu().numpy() + \
                                pred[:, j] * target_mask[0, :, j].detach().cpu().numpy()
                    visual(true[:, j], filled, os.path.join(folder_path, str(epoch) + '_' + str(j) + '.png'))
            
            if self.local_rank == 0:
                train_end_time = time.time()
                print("Epoch: {} training time cost: {}".format(epoch + 1, train_end_time - epoch_time))
            
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)
            dist.reduce(train_loss, 0, op=dist.ReduceOp.SUM) # sum loss from all gpus
            dist.reduce(vali_loss, 0, op=dist.ReduceOp.SUM)
            dist.reduce(test_loss, 0, op=dist.ReduceOp.SUM)
            dist.reduce(train_size, 0, op=dist.ReduceOp.SUM)
            dist.reduce(vali_size, 0, op=dist.ReduceOp.SUM)
            dist.reduce(test_size, 0, op=dist.ReduceOp.SUM)

            if self.local_rank == 0:
                print("Epoch: {0}, eval time cost: {1:.2f}, Steps: {2} | Train Loss: {3:.7f} Vali Loss: {4:.7f} Test Loss: {5:.7f}".format(
                    epoch + 1, 
                    time.time()-train_end_time, 
                    train_steps, 
                    train_loss / train_size, 
                    vali_loss / vali_size, 
                    test_loss / test_size
                    ))
                
                early_stopping(vali_loss / vali_size, self.model, model_optim, scheduler, path)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

            scheduler.step(train_loss)

        dist.destroy_process_group()
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0, pred_loader=None):
        if pred_loader is None:
            test_data, test_loader = self._get_data(flag='test')
        else:
            test_loader = pred_loader
            test_data, _ = self._get_data(flag='test')

        _, K = test_data[0][0].shape
        mask = self.create_fixed_mask(K)
        mask = torch.from_numpy(np.repeat(mask[np.newaxis, :, :], test_loader.batch_size, axis=0))

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))


        preds = []
        trues = []
        target_masks = []
        folder_path = './test_results/' + setting + '/'

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, actual_mask, _) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                # random mask
                # B, T, N = batch_x.shape
                # mask = torch.rand((B, T, N)).to(self.device)
                # mask[mask <= self.args.mask_rate] = 0  # masked
                # mask[mask > self.args.mask_rate] = 1  # remained
                mask = mask.to(self.device)
                if type(actual_mask) is not int:
                    actual_mask = actual_mask.to(self.device)
                    target_mask = actual_mask - mask
                    target_mask[target_mask <0] = 0
                else:
                    target_mask = 1 - mask
                    actual_mask = 1

                inp = batch_x.masked_fill(mask == 0, 0)

                # imputation
                outputs = self.model(inp, batch_x_mark, None, None, mask, _)

                # eval
                f_dim = 0
                outputs = outputs[:, :, f_dim:]
                outputs = outputs.detach().cpu().numpy()
                pred = outputs
                true = batch_x.detach().cpu().numpy()

                pred = test_data.inverse_transform(pred[0]).reshape(pred.shape)
                true = test_data.inverse_transform(true[0]).reshape(true.shape)

                preds.append(pred)
                trues.append(true)
                target_masks.append(target_mask.detach().cpu())

                if i % 20 == 0:
                    filled = true[0, :, -1].copy()
                    filled = filled * (actual_mask-target_mask)[0, :, -1].detach().cpu().numpy() + \
                                pred[0, :, -1] * target_mask[0, :, -1].detach().cpu().numpy()
                    visual(true[0, :, -1], filled, os.path.join(folder_path, str(i) + '.png'))

        preds = np.concatenate(preds, 0)
        trues = np.concatenate(trues, 0)
        target_masks = np.concatenate(target_masks, 0)
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds[target_masks == 1], trues[target_masks == 1])
        print('rmse:{:.2f}, mae:{:.2f}, mape:{:.2f}%'.format(rmse, mae, mape*100))
        f = open("result_imputation.txt", 'a')
        f.write(setting + "  \n")
        f.write('rmse:{:.2f}, mae:{:.2f}, mape:{:.2f}%'.format(rmse, mae, mape*100))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        return
    
    def pred(self, setting):
        test_data, _ = self._get_data(flag='test')
        _, pred_loader = self._get_data(flag='pred', scaler=test_data.pred_transform())

        self.test(setting, test=1, pred_loader=pred_loader)
    
    def create_fixed_mask(self,dim_k):
        # Define the start and end times of each range
        ranges_5min = [('05:00:00', '17:55:00')]

        # Define the date for which to generate the DatetimeIndex values
        target_date = '2023-07-01'

        # Create an empty list to store the DatetimeIndex values
        index_values = []

        # Loop through each range and generate the DatetimeIndex values within that range for the target date
        for start_time, end_time in ranges_5min:
            index_values += pd.date_range(start=f'{target_date} {start_time}', end=f'{target_date} {end_time}', freq='5min', inclusive='both').tolist()

        # Create the DatetimeIndex from the list of values
        ranges_5min = pd.DatetimeIndex(index_values)
        empty_df = pd.DataFrame(index=ranges_5min, columns=[str(i) for i in range(0, dim_k)])

        ranges_5min_obs = [('05:00:00', '07:55:00'), ('09:30:00', '12:25:00'), ('14:00:00', '16:55:00')]

        # Define the date for which to generate the DatetimeIndex values
        target_date = '2023-07-01'

        # Create an empty list to store the DatetimeIndex values
        index_values = []

        # Loop through each range and generate the DatetimeIndex values within that range for the target date
        for start_time, end_time in ranges_5min_obs:
            index_values += pd.date_range(start=f'{target_date} {start_time}', end=f'{target_date} {end_time}', freq='5min', inclusive='both').tolist()

        ranges_5min_obs = pd.DatetimeIndex(index_values)

        empty_df.loc[ranges_5min_obs, :] = 1
        empty_df.fillna(0, inplace=True)

        return empty_df.values