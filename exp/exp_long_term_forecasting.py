from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw,accelerated_dtw
from utils.dilate import Dilate
from utils.augmentation import run_augmentation,run_augmentation_single
from utils.losses import Loss, ReprLoss, LearnableNetwork
from utils.pc_model import Model, PC_Model
#import wandb

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        if args.use_ph:
            self.get_topo_feature = PC_Model(args).float()

        if args.adversarial:
            self.discriminator = self._build_discriminator()

        #if args.use_wandb:
            #self.wandb = wandb
            #self.wandb.init(project='loss function', config={
                #"task": args.task_name,
                #"model_id": args.model_id,
                #"model": args.model,
                #"features": args.features,
                #"training_epochs": args.train_epochs,
                #"batch_size": args.batch_size,
                #"loss": args.loss,
            #})
        
        #if args.loss == "MSE":
            #self.wandb.run.name = f"{args.task_name}_{args.model_id}_{args.model}_{args.features}_{args.train_epochs}_{args.batch_size}_{args.loss}"
        #elif args.loss == "combined_loss":
            #self.wandb.run.name = f"{args.task_name}_{args.model_id}_{args.model}_{args.features}_{args.train_epochs}_{args.batch_size}_{args.loss}"
        #elif args.loss == "learned_repr_loss":
            #self.wandb.run.name = f"{args.task_name}_{args.model_id}_{args.model}_{args.features}_{args.train_epochs}_{args.batch_size}_{args.loss}"

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    # def _build_learnable_network(self):
    #     """
    #     Build the loss network using the LearnableNetwork class.
    #     """
    #     repr_network = LearnableNetwork(self.args).repr_network.to(self.device)
    #     return repr_network

    def _build_discriminator(self):
        """
        Build the discriminator using the LearnableNetwork class.
        """
        discriminator = LearnableNetwork(self.args).discriminator.to(self.device)
        return discriminator

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        if self.args.adversarial:
            #repr_network_optim = optim.Adam(self.repr_network.parameters(), lr=self.args.learning_rate)
            discriminator_optim = optim.Adam(self.discriminator.parameters(), lr=self.args.learning_rate)
            return model_optim, discriminator_optim
        else:
            return model_optim

    def _select_criterion(self):
        if self.args.additional is None:
            if self.args.base == "MSE":
                criterion_base = nn.MSELoss()
            elif self.args.base == "MAE":
                criterion_base = nn.L1Loss()
            
            if self.args.repr_loss: # representation loss option
                criterion_train_new = ReprLoss(self.args)
            elif self.args.adversarial:
                criterion_train_new = LearnableNetwork(self.args)
            else:
                criterion_train_new = None
        
        else: # additional loss options
            criterion_base = nn.MSELoss() # only for vali and test loss calculation
            criterion_train_new = Loss(self.args).to(self.device)
        
        # new vali metric
        if self.args.vali_metric == "dilate":
            criterion_vali_new = Dilate()
        else:
            criterion_vali_new = None

        return criterion_base, criterion_train_new, criterion_vali_new

    def vali(self, vali_data, vali_loader, criterion, test):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.use_ph:
                            batch_y_ph = batch_y[:, -self.args.pred_len:, :].to(self.device)
                            topo_y = self.get_topo_feature(batch_y_ph) 
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, topo_y)
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.use_ph:
                        batch_y_ph = batch_y[:, -self.args.pred_len:, :].to(self.device)
                        topo_y = self.get_topo_feature(batch_y_ph) 
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, topo_y)
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                if self.args.vali_metric == "dilate" and test == 0:
                    loss, _ , _ = criterion.dilate_metric(pred, true, self.args.device, self.args.alpha_dilate, self.args.gamma_dilate, self.args.batch_size)
                else:
                    loss = criterion(pred, true)
                
                total_loss.append(loss)
        total_loss = np.average(total_loss)
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

        if self.args.adversarial:
            model_optim, discriminator_optim = self._select_optimizer()
        else:
            model_optim = self._select_optimizer()
        criterion_base, criterion_train_new, criterion_vali_new = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            if self.args.adversarial:
                MSE_loss = []
                loss_F = []
                loss_D = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.use_ph:
                            batch_y_ph = batch_y[:, -self.args.pred_len:, :].to(self.device)
                            topo_y = self.get_topo_feature(batch_y_ph)
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, topo_y)
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                            # outputs dimension: [batch_size, pred_len, num_features]

                        f_dim = -1 if self.args.features == 'MS' else 0

                        if self.args.additional is not None:
                            if self.args.include_input_range:
                                outputs = torch.cat([batch_y[:, :self.args.label_len, :], outputs], dim=1).float().to(self.device)
                                outputs = outputs[:, :, f_dim:]
                                batch_y = batch_y[:, :, f_dim:].to(self.device)
                                loss = criterion_train_new(outputs, batch_y)

                                train_loss.append(loss.item())
                            else:
                                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                                loss = criterion_train_new(outputs, batch_y)

                                train_loss.append(loss.item())
                        else:
                            outputs = outputs[:, -self.args.pred_len:, f_dim:]
                            batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                            if self.args.repr_loss:
                                loss = criterion_train_new(batch_x, outputs, batch_y)
                                train_loss.append(loss.item())
                            
                            elif self.args.adversarial:
                                mse_loss = criterion_base(outputs, batch_y)
                                MSE_loss.append(mse_loss.item())

                                loss_f, loss_d = criterion_train_new(batch_x, outputs, batch_y)
                                loss_F.append(loss_f.item())
                                loss_D.append(loss_d.item())

                                loss = mse_loss + self.args.lambda_F * loss_f
                                train_loss.append(loss.item())
                            
                            else:
                                loss = criterion_base(outputs, batch_y)
                                train_loss.append(loss.item())

                else:
                    if self.args.use_ph:
                        batch_y_ph = batch_y[:, -self.args.pred_len:, :].to(self.device)
                        topo_y = self.get_topo_feature(batch_y_ph)
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, topo_y)
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0

                    if self.args.additional is not None:
                        if self.args.include_input_range:
                            outputs = torch.cat([batch_y[:, :self.args.label_len, :], outputs], dim=1).float().to(self.device)
                            outputs = outputs[:, :, f_dim:]
                            batch_y = batch_y[:, :, f_dim:].to(self.device)
                            loss = criterion_train_new(outputs, batch_y)

                            train_loss.append(loss.item())
                        else:
                            outputs = outputs[:, -self.args.pred_len:, f_dim:]
                            batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                            loss = criterion_train_new(outputs, batch_y)

                            train_loss.append(loss.item())
                    else:
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                        if self.args.repr_loss:
                            loss = criterion_train_new(batch_x, outputs, batch_y)
                            train_loss.append(loss.item())
                        
                        elif self.args.adversarial:
                            mse_loss = criterion_base(outputs, batch_y)
                            MSE_loss.append(mse_loss.item())

                            loss_f, loss_d = criterion_train_new(batch_x, outputs, batch_y)
                            loss_F.append(loss_f.item())
                            loss_D.append(loss_d.item())

                            loss = mse_loss + self.args.lambda_F * loss_f
                            train_loss.append(loss.item())
                        
                        else:
                            loss = criterion_base(outputs, batch_y)
                            train_loss.append(loss.item())

                        
                if (i + 1) % 100 == 0:
                    if self.args.adversarial:
                        print("\titers: {0}, epoch: {1} | mse_loss: {2:.7f} | loss_F: {3:.7f} | loss_D: {4:.7f}".format(i + 1, epoch + 1, mse_loss.item(), loss_f.item(), loss_d.item()))
                    else:
                        print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    if self.args.adversarial:
                        discriminator_optim.zero_grad()
                        loss_d.backward()
                        discriminator_optim.step()
                        
                        model_optim.zero_grad()
                        loss.backward()
                        model_optim.step()
                    else:
                        loss.backward()
                        model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            
            train_loss = np.average(train_loss)
            
            if self.args.vali_metric is not None:
               vali_loss = self.vali(vali_data, vali_loader, criterion_vali_new, test=0)
            else:
               vali_loss = self.vali(vali_data, vali_loader, criterion_base, test=0)

            test_loss = self.vali(test_data, test_loader, criterion_base, test=1)

            # if self.args.wandb:
            #     self.wandb.log({
            #         "train_loss": train_loss,
            #         "vali_loss": vali_loss,
            #         "test_loss": test_loss, 
            #     })

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        inputs = []

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.use_ph:
                            topo_y = self.get_topo_feature(batch_y)
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, topo_y)
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.use_ph:
                        topo_y = self.get_topo_feature(batch_y)
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, topo_y)
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                #batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = batch_y.shape
                    if self.args.features == 'MS':
                        outputs = np.tile(outputs, [1, 1, batch_y.shape[-1]])
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)
        
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                # input
                input = batch_x.detach().cpu().numpy()
                inputs.append(input)

                preds.append(pred)
                trues.append(true)
                if i % 10 == 0:
                    #input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        inputs = np.concatenate(inputs, axis=0)
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)
        print("inputs shape:", inputs.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1,1)
                y = trues[i].reshape(-1,1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = 'not calculated'
        
        if self.args.use_dilate and self.args.use_gpu:
            dilate_metrics, shape_metrics, temporal_metrics = Dilate().dilate_metric(preds, trues, self.args.device, self.args.alpha_dilate, self.args.gamma_dilate, batch_size=1)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'inputs.npy', inputs)
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        if self.args.use_dilate:
            np.save(folder_path + 'dilate_metrics.npy', np.array(dilate_metrics))
            np.save(folder_path + 'shape_metrics.npy', np.array(shape_metrics))
            np.save(folder_path + 'temporal_metrics.npy', np.array(temporal_metrics))

        return
