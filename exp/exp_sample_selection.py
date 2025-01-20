from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import visual_sample
import os
import torch

class Exp_Sample_Selection(Exp_Basic):
    def __init__(self, args):
        super(Exp_Sample_Selection, self).__init__(args)
    
    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model
        
    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader
    
    def train_data_selection(self):
        train_data, train_loader = self._get_data(flag='train')

        path = os.path.join(self.args.sample_selection, 'train')
        if not os.path.exists(path):
            os.makedirs(path)
        

        iter_count = 0
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            iter_count += 1

            for j in range(batch_x.shape[0]):
                # inputs
                inputs = batch_x[j, :, -1].float()
                
                # targets
                batch_y = batch_y.float()
                targets = batch_y[j, -(self.args.pred_len+1):, -1]
                
                # visualize
                visual_sample(inputs, targets, os.path.join(path, f'{i}_{j}.pdf'))
    
    
    def test_data_selection(self):
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.sample_selection, 'test')
        if not os.path.exists(path):
            os.makedirs(path)
        

        iter_count = 0
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            iter_count += 1

            for j in range(batch_x.shape[0]):
                # inputs
                inputs = batch_x[j, :, -1].float()
                
                # targets
                batch_y = batch_y.float()
                targets = batch_y[j, -(self.args.pred_len+1):, -1]
                
                # visualize
                visual_sample(inputs, targets, os.path.join(path, f'{i}_{j}.pdf'))