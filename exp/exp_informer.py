from data.data_loader import Dataset_Custom, Dataset_Pred
from exp.exp_basic import Exp_Basic
from models.model import Informer, InformerStack, decomposition_nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric,metric1,metric3
from utils.metrics1 import metricX
from utils.metricX import metricB

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import os
import time

import warnings
warnings.filterwarnings('ignore')
from torch.autograd import Variable

plt.switch_backend('agg')

class CustomLoss(nn.Module):
    def __init__(self, initial_lambda=0.005):
        super(CustomLoss, self).__init__()
        # Define lambda as a learnable parameter
        self.lamda = nn.Parameter(torch.tensor(initial_lambda, requires_grad=True))
    
    def forward(self, true, pred, gc_adjusted_sum):
        # Calculate the base loss (e.g., Mean Squared Error)
        base_loss = F.mse_loss(pred, true)
        
        # Calculate the total loss
        total_loss = base_loss + gc_adjusted_sum*self.lamda
        
        return total_loss

class Exp_Informer(Exp_Basic):
    def __init__(self, args):
        super(Exp_Informer, self).__init__(args)
    
    def _build_model(self):
        model_dict = {
            'informer':Informer,
            'informerstack':InformerStack,
            'decomposition':decomposition_nn
        }
        if self.args.model=='informer' or self.args.model=='informerstack' or self.args.model=='decomposition':
            e_layers = self.args.e_layers if self.args.model=='informer' or self.args.model=='informerstack' or self.args.model=='informer' or self.args.model=='decomposition' else self.args.s_layers
            model = model_dict[self.args.model](
                self.args.enc_in,
                self.args.dec_in, 
                self.args.c_out, 
                self.args.seq_len, 
                self.args.label_len,
                self.args.pred_len, 
                self.args.factor,
                self.args.d_model, 
                self.args.n_heads, 
                e_layers, # self.args.e_layers,
                self.args.d_layers, 
                self.args.d_ff,
                self.args.dropout, 
                self.args.attn,
                self.args.embed,
                self.args.freq,
                self.args.activation,
                self.args.output_attention,
                self.args.distil,
                self.args.mix,
                self.device
            ).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        args = self.args

        data_dict = {
            'HPC':Dataset_Custom,
        }
        Data = data_dict[self.args.data]
        timeenc = 0 if args.embed!='timeF' else 1

        if flag == 'test':
            shuffle_flag = False; drop_last = True; batch_size = args.batch_size; freq=args.freq
        elif flag=='pred':
            shuffle_flag = False; drop_last = False; batch_size = 1; freq=args.detail_freq
            Data = Dataset_Pred
        else:
            shuffle_flag = True; drop_last = True; batch_size = args.batch_size; freq=args.freq
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            cols=args.cols
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    def _select_criterion(self):
        # criterion = nn.L1Loss()
        # criterion =nn.MSELoss()
        # loss_function = calculate_imse()
        criterion = CustomLoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(vali_loader):
            train, pred, true,gc_adjusted_sum,adp  = self._process_one_batch(
                vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            # pred = pred /
            # print(gc_adjusted_sum)
            loss = criterion(pred.detach().cpu(), true.detach().cpu(),gc_adjusted_sum)
            # print(true.detach().cpu()[:,:,-1:].shape)
            # loss = criterion(true[:,:,-1:].flatten(),true[:,:,-2:-1].flatten(),pred[:,:,-1:].flatten(),pred[:,:,-2:-1].flatten())
            # loss = criterion(true.detach().cpu()[:,:,0].flatten(),true.detach().cpu()[:,:,1].flatten(),pred.detach().cpu()[:,:,0].flatten(),pred.detach().cpu()[:,:,1].flatten())
            # loss = criterion((pred), (true))
            # /0.39147269
            # loss = criterion(pred.detach().cpu()[:,:,-6:].sum(axis=2), true.detach().cpu()[:,:,-6:].sum(axis=2))
            
            total_loss.append(loss.detach().cpu())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag = 'train')
        vali_data, vali_loader = self._get_data(flag = 'val')
        test_data, test_loader = self._get_data(flag = 'test')
        # 迭代查看前几个数据
        num_examples = 1  # 要查看的数据数量

        # 使用enumerate函数遍历dataloader，获取数据和对应的标签
        for i, (seq_x, seq_y, seq_x_mark, seq_y_mark) in enumerate(train_loader):
            if i >= num_examples:
                break
            
            # 打印数据和标签
            print(f"Data: {seq_y}")

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        model_optim = self._select_optimizer()
        criterion =  self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                
                model_optim.zero_grad()
                train, pred, true, gc_adjusted_sum,adp = self._process_one_batch(
                    train_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
                loss = criterion((pred), (true), gc_adjusted_sum)
                # loss = criterion(true.detach().cpu()[:,:,0].flatten(),true.detach().cpu()[:,:,1].flatten(),pred.detach().cpu()[:,:,0].flatten(),pred.detach().cpu()[:,:,1].flatten())
                # loss = criterion(true[:,:,-1:].flatten(),true[:,:,-2:-1].flatten(),pred[:,:,-1:].flatten(),pred[:,:,-2:-1].flatten())
                # /0.15765888
                # loss = criterion(true.detach().cpu()[:,:,-1:].flatten(),true.detach().cpu()[:,:,-2:-1].flatten(),pred.detach().cpu()[:,:,-1:].flatten(),pred.detach().cpu()[:,:,-2:-1].flatten())
            # loss = criterion((pred), (true))
                # loss = criterion(pred[:,:,-6:].sum(axis=2), true[:,:,-6:].sum(axis=2))
                train_loss.append(loss.item())
                if (i+1) % 100==0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                    print(gc_adjusted_sum)
                    adp_np = adp.detach().cpu().numpy()  # 将 adp 转换为 numpy 数组
                    adp_df = pd.DataFrame(adp_np)        # 转换为 pandas DataFrame
                    csv_filename = f"adp_{i+1}.csv"      # 按照 i 命名文件
                    adp_df.to_csv(csv_filename, index=False)  # 保存为 CSV 文件
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch+1, self.args)
            
        best_model_path = path+'/'+'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        
        return self.model

    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        self.model.eval()
        
        preds = []
        trues = []
        trains = []
        
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(test_loader):
            # print('batch_y:',batch_y)
            train, pred, true, gc_adjusted_sum,adp = self._process_one_batch(
                test_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            # preds.append(pred.detach().cpu().numpy()[:,:,-6:].sum(axis=2))
            # pred = pred / (-1.89193278)
            preds.append(pred.detach().cpu().numpy())
            # (-1.00580797)
            # trues.append(true.detach().cpu().numpy()[:,:,-6:].sum(axis=2))
            trues.append(true.detach().cpu().numpy())
            trains.append(train.detach().cpu().numpy())
            if i % 10 == 0:
                input = batch_x.detach().cpu().numpy()
                gt = np.concatenate((input[0, :, -1], true.detach().cpu().numpy()[0, :, -1]), axis=0)
                pd = np.concatenate((input[0, :, -1], pred.detach().cpu().numpy()[0, :, -1]), axis=0)
                visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        trains = np.array(trains)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        # preds = np.abs(preds)
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        preds = torch.tensor(preds, dtype=torch.float32)
        trues = torch.tensor(trues, dtype=torch.float32)
        mse = F.mse_loss(preds,trues)
        print(mse)
        # imae, imse, imspe, ui, arv = metricB(preds, trues)
        # print('imae:{}, imse:{}, imspe:{}, ui:{}, arv:{}'.format(imae, imse, imspe, ui, arv))
        # mae, mse, rmse, mape, mspe, rse, corr, nd, nrmse = metricX(preds, trues)
        # print('nd:{}, nrmse:{}, mse:{}, mae:{}, rse:{}, mape:{},corr:{}'.format(nd, nrmse,mse, mae, rse, mape,corr))
        # print(pred.cpu().data.numpy().shape)
        # mae,mse,rmse,mape,mspe,r2 = metric(pred.cpu().data.numpy(),true.cpu().data.numpy())
        # print('mae,mse,rmse,mape,mspe,r2')
        # print('mae+{}+mse+{}+rmse+{}+mape+{}+mspe+{}+r2+{}'.format(mae,mse,rmse,mape,mspe,r2))
        # imae,imse,imspe,isse= metric1(pred.cpu().data.numpy(),true.cpu().data.numpy())
        # # 假设preds和trues都是形状为(m, n)的numpy数组，其中m是样本数量，n是特征数量
        # m = preds.shape[0]

        # # 创建一个空的DataFrame
        # df = pd.DataFrame()

        # # 遍历每个样本
        # for i in range(m):
        #     # 获取第i个样本的预测值和真实值
        #     pred = list(preds[i,:])
        #     true = list(trues[i,:])
            
        #     # 创建一个临时DataFrame，其中包含两列分别为预测值和真实值
        #     temp_df = pd.DataFrame({'Prediction': pred, 'True': true})
            
        #     # 将临时DataFrame追加到主DataFrame中
        #     df = df.append(temp_df, ignore_index=True)

        # # 将DataFrame导出为CSV文件，文件名为'predictions.csv'
        # df.to_csv(folder_path+'predictions.csv', index=False)
        # print(preds[0, :])
        # print(trues[0, :])
        # for i in range(20):
        #   df_preds_l = pd.DataFrame(preds[i, :])
        #   df_trues = pd.DataFrame(trues[i, :])

        #   # 绘制图像
        #   plt.figure(figsize=(10, 5))
        #   plt.plot(df_preds, label='Predicted')
        #   plt.plot(df_trues, label='True')
        #   plt.legend()
        #   plt.xlabel('Index')
        #   plt.ylabel('Value')
        #   plt.title('Predicted vs True Values')
        #   # 保存图片
        #   file_path = os.path.join(folder_path, f'predicted_vs_true_{i}.png')
        #   plt.savefig(file_path)
        # print('imae:{},imse:{},imspe:{},isse:{}'.format(imae,imse,imspe,isse))
        # np.save(folder_path+'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path+'pred.npy', preds)
        np.save(folder_path+'true.npy', trues)
        np.save(folder_path+'train.npy', trains)

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')
        print(1234567)
        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path+'/'+'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        self.model.eval()
        
        preds = []
        
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(pred_loader):
            train, pred, true = self._process_one_batch(
                pred_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            preds.append(pred.detach().cpu().numpy())
        preds = np.array(preds)
        # print(preds.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        mse = F.mse_loss(preds,true.cpu().data.numpy())
        print(mse)
        from sklearn.metrics import mean_absolute_error, r2_score
        mae = mean_absolute_error(true, preds)
# R²
        r2 = r2_score(true, preds)
        mae = F.L1Loss(preds,true.cpu().data.numpy())
        print(mae)
        print(r2)

        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        np.save(folder_path+'real_prediction.npy', preds)
        np.save(folder_path+'real_true.npy', true.cpu().data.numpy())
        print('over')
        return

    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float()

        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)
        # print(1)
        # decoder input
        if self.args.padding==0:
            dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        elif self.args.padding==1:
            dec_inp = torch.ones([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        dec_inp = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).float().to(self.device)
        # encoder - decoder
        # print(11)
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                if self.args.output_attention:
                    # outputs = self.model(batch_x)
                    outputs,gc_adjusted_sum = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs,gc_adjusted_sum = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            if self.args.output_attention:
                outputs,gc_adjusted_sum,adp = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                # , batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs,gc_adjusted_sum,adp = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                #  outputs = self.model(batch_x)
        # print(type(outputs))
        # print(outputs.shape) 
        # print(111)
        if self.args.inverse:
            outputs = dataset_object.inverse_transform(outputs)
        f_dim = -1 if self.args.features=='MS' else 0
        batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)
        # print("outputs",outputs)
        # print("batch_y",batch_y)
#         <class 'torch.Tensor'>
# torch.Size([512, 96, 13])
        return dataset_object.inverse_transform(batch_x), outputs, batch_y, gc_adjusted_sum,adp
