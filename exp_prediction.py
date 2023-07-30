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


class Exp_Prediction(Exp_Basic):
    def __init__(self, args):
        super(Exp_Prediction, self).__init__(args)

    def _build_model(self):

        if self.args.use_multi_gpu and self.args.use_gpu:
            dist.init_process_group(backend='nccl')
            self.local_rank = int(os.environ['LOCAL_RANK'])
            torch.cuda.set_device(self.local_rank)
            model = self.model_dict[self.args.model].Model(self.args).cuda()
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = nn.parallel.DistributedDataParallel(model, device_ids=[self.local_rank])
            self.scaler = GradScaler() # mixed precision training
        return model

    def _get_data(self, flag, scaler=None):
        data_set, data_loader = data_provider(self.args, flag, scaler)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=1e-5)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss().cuda()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = torch.tensor(0.).cuda()
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, _, actual_mask) in enumerate(vali_loader):
                batch_x = batch_x.float().cuda()
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().cuda()
                batch_y_mark = batch_y_mark.float().cuda()

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().cuda()


                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].cuda()

                pred = outputs
                true = batch_y

                loss = criterion(pred[actual_mask==1], true[actual_mask==1])

                total_loss += loss

        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        folder_path = './train_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            model_optim, 
            factor=0.8, 
            patience=5, 
            verbose=True
            )
        

        for epoch in range(self.args.train_epochs):
            iter_count = 0

            train_loss = torch.tensor(0.).cuda()
            train_size = torch.tensor(len(train_loader)).float().cuda()
            vali_size = torch.tensor(len(vali_loader)).float().cuda()
            test_size = torch.tensor(len(test_loader)).float().cuda()

            train_loader.sampler.set_epoch(epoch) # prevent sampling bug
            self.model.train()
            epoch_time = time.time()
            # here actual_mask is mask for y seq only, where 0 values in the data mask as 0, otherwise 1
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, _, actual_mask) in enumerate(train_loader):
                iter_count += 1

                batch_x = batch_x.float().cuda()
                batch_y = batch_y.float().cuda()
                batch_x_mark = batch_x_mark.float().cuda()
                batch_y_mark = batch_y_mark.float().cuda()

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().cuda()


                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim =  0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].cuda()
                loss = criterion(outputs[actual_mask==1], batch_y[actual_mask==1])
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
                pred = outputs[0] # (L_pred, K)
                true = batch_y.detach().cpu().numpy()[0] # (L_pred, K)
                mask = actual_mask.detach().cpu().numpy()[0] # (L_pred, K)

                pred = train_data.inverse_transform(pred)
                true = train_data.inverse_transform(true)
                hist = train_data.inverse_transform(batch_x.detach().cpu().numpy()[0])

                for j in range(true.shape[1]):
                    gt = np.concatenate((hist[:, j], true[:, j]*mask[:,j]), axis=0)
                    pd = np.concatenate((hist[:, j], pred[:, j]*mask[:,j]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(epoch) + '_' + str(j) + '.png'))
            
            if self.local_rank == 0:
                print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)
            dist.reduce(train_loss, 0, op=dist.ReduceOp.SUM) # sum loss from all gpus
            dist.reduce(vali_loss, 0, op=dist.ReduceOp.SUM)
            dist.reduce(test_loss, 0, op=dist.ReduceOp.SUM)
            dist.reduce(train_size, 0, op=dist.ReduceOp.SUM)
            dist.reduce(vali_size, 0, op=dist.ReduceOp.SUM)
            dist.reduce(test_size, 0, op=dist.ReduceOp.SUM)

            if self.local_rank == 0:
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss / train_size, vali_loss / vali_size, test_loss / test_size))

                early_stopping(vali_loss / vali_size, self.model, path)
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

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))


        preds = []
        trues = []
        masks = []

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, _, actual_mask) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)


                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                if type(actual_mask) is int:
                    actual_mask = torch.ones_like(outputs)

                # eval
                f_dim = 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs
                true = batch_y
                mask = actual_mask.detach().cpu().numpy()[0] # (L_pred, K)

                # batch_size of testing has to be 1
                pred = test_data.inverse_transform(pred[0])
                true = test_data.inverse_transform(true[0])
                hist = test_data.inverse_transform(batch_x.detach().cpu().numpy()[0])

                preds.append(pred)
                trues.append(true)
                masks.append(mask)

                if i % 20 == 0:
                    gt = np.concatenate((hist[:, -1], true[:, -1]*mask[:,-1]), axis=0)
                    pd = np.concatenate((hist[:, -1], pred[:, -1]*mask[:,-1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.png'))

        preds = np.array(preds)
        trues = np.array(trues)
        masks = np.array(masks)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds[masks==1], trues[masks==1])
        print('rmse:{:.2f}, mae:{:.2f}, mape:{:.2f}%'.format(rmse, mae, mape*100))
        f = open("result_prediction.txt", 'a')
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