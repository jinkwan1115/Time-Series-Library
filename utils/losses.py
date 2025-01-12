# This source code is provided for the purposes of scientific reproducibility
# under the following limited license from Element AI Inc. The code is an
# implementation of the N-BEATS model (Oreshkin et al., N-BEATS: Neural basis
# expansion analysis for interpretable time series forecasting,
# https://arxiv.org/abs/1905.10437). The copyright to the source code is
# licensed under the Creative Commons - Attribution-NonCommercial 4.0
# International license (CC BY-NC 4.0):
# https://creativecommons.org/licenses/by-nc/4.0/.  Any commercial use (whether
# for the benefit of third parties or internally in production) requires an
# explicit license. The subject-matter of the N-BEATS model and associated
# materials are the property of Element AI Inc. and may be subject to patent
# protection. No license to patents is granted hereunder (whether express or
# implied). Copyright © 2020 Element AI Inc. All rights reserved.

"""
Loss functions for PyTorch.
"""

import torch as t
import torch.nn as nn
import numpy as np
import pdb

from utils.polynomials import laguerre_torch

def divide_no_nan(a, b):
    """
    a/b where the resulted NaN or Inf are replaced by 0.
    """
    result = a / b
    result[result != result] = .0
    result[result == np.inf] = .0
    return result


class mape_loss(nn.Module):
    def __init__(self):
        super(mape_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        MAPE loss as defined in: https://en.wikipedia.org/wiki/Mean_absolute_percentage_error

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        weights = divide_no_nan(mask, target)
        return t.mean(t.abs((forecast - target) * weights))


class smape_loss(nn.Module):
    def __init__(self):
        super(smape_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        sMAPE loss as defined in https://robjhyndman.com/hyndsight/smape/ (Makridakis 1993)

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        return 200 * t.mean(divide_no_nan(t.abs(forecast - target),
                                          t.abs(forecast.data) + t.abs(target.data)) * mask)


class mase_loss(nn.Module):
    def __init__(self):
        super(mase_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        MASE loss as defined in "Scaled Errors" https://robjhyndman.com/papers/mase.pdf

        :param insample: Insample values. Shape: batch, time_i
        :param freq: Frequency value
        :param forecast: Forecast values. Shape: batch, time_o
        :param target: Target values. Shape: batch, time_o
        :param mask: 0/1 mask. Shape: batch, time_o
        :return: Loss value
        """
        masep = t.mean(t.abs(insample[:, freq:] - insample[:, :-freq]), dim=1)
        masked_masep_inv = divide_no_nan(mask, masep[:, None])
        return t.mean(t.abs(target - forecast) * masked_masep_inv)

class FrequencyLoss(nn.Module):
    def __init__(self, args):
        super(FrequencyLoss, self).__init__()
        self.args = args
    def forward(self, predictions, targets):
        # Fourier Transform
        pred_fft = t.fft.rfft(predictions, dim=1)
        target_fft = t.fft.rfft(targets, dim=1)

        # Frequency Loss - Complex
        if self.args.freq_loss_type == 'complex':
            
            freq_loss = pred_fft - target_fft

            # if self.args.loss == 'MSE':
            #     freq_loss = t.mean(t.abs(freq_loss) ** 2)

            #if self.args.loss == 'MAE':
            freq_loss = t.mean(t.abs(freq_loss))

        # Frequency Loss - Magnitude
        if self.args.freq_loss_type == 'mag':

            pred_fft_mag = t.abs(pred_fft)
            target_fft_mag = t.abs(target_fft)

            freq_loss = pred_fft_mag - target_fft_mag

            # if self.args.loss == 'MSE': 
            #     freq_loss = t.mean(t.abs(freq_loss) ** 2)

            #if self.args.loss == 'MAE':
            freq_loss = t.mean(t.abs(freq_loss))
        
        # Frequency Loss - Phase
        if self.args.freq_loss_type == 'phase':

            pred_fft_phase = t.angle(pred_fft)
            target_fft_phase = t.angle(target_fft)

            freq_loss = pred_fft_phase - target_fft_phase

            # if self.args.loss == 'MSE':     
            #     freq_loss = t.mean(t.abs(freq_loss) ** 2)

            #if self.args.loss == 'MAE':
            freq_loss = t.mean(t.abs(freq_loss))

        return freq_loss

class LaguerreLoss(nn.Module):
    def __init__(self, args):
        super(LaguerreLoss, self).__init__()
        self.degree = args.degree
        self.device = args.device

    def forward(self, predictions, targets):
        pred_laguerre = laguerre_torch(predictions, degree=self.degree, rtn_data=False, device=self.device)
        target_laguerre = laguerre_torch(targets, degree=self.degree, rtn_data=False, device=self.device)
        
        laguerre_loss = t.mean(t.abs(pred_laguerre - target_laguerre))

        return laguerre_loss

class CombinedLoss(nn.Module):
    def __init__(self, args):
        super(CombinedLoss, self).__init__()
        self.args = args
        self.alpha = args.alpha_additional_loss
        self.mse_loss = nn.MSELoss()
        self.freq_loss = FrequencyLoss(args)
        self.laguerre_loss = LaguerreLoss(args)
    
    def forward(self, predictions, targets):
        mse_loss = self.mse_loss(predictions, targets)
        if self.args.use_laguerre:
            additional_loss = self.laguerre_loss(predictions, targets)
        if self.args.use_freq:
            additional_loss = self.freq_loss(predictions, targets)

        return self.alpha * additional_loss + (1 - self.alpha) * mse_loss
