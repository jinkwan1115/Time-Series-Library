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
Various Loss functions for Time Series Analysis
"""

import torch as t
import torch.nn as nn
import numpy as np
import pdb

from utils.polynomials import laguerre_torch, hermite_torch, legendre_torch, chebyshev_torch
from utils.cka import cka_torch

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


# jinmyeong
class WindowShapeLoss(nn.Module):
    def __init__(self, base, distance="EM", temp_to="both", temp=0.01):
        super(WindowShapeLoss, self).__init__()
        self.distance = distance
        if temp_to == "both":
            self.temp_p, self.temp_q = temp, temp
        elif temp_to == "true":
            self.temp_p, self.temp_q = 1, temp
        else:
            self.temp_p, self.temp_q = temp, 1

    def dist_loss(self, predictions, targets):
        p = t.softmax(predictions/self.temp_p, -1) + 1e-8 # epsilon(1e-8) is to avoid division by zero
        q = t.softmax(targets/self.temp_q, -1) + 1e-8 # epsilon(1e-8) is to avoid division by zero
        if self.distance == "KL":
            distance = t.mean(t.sum(p * t.log(p / q)), dim=0)
        elif self.distance == "EM":
            cdf_p = t.cumsum(p, dim=-1)
            cdf_q = t.cumsum(q, dim=-1)
            distance = t.mean(t.abs(cdf_p - cdf_q))
        return distance

    def forward(self, predictions, targets):
        return self.dist_loss(predictions, targets)


# FreDF (Fourier Transform)
class FrequencyLoss(nn.Module):
    def __init__(self, args):
        super(FrequencyLoss, self).__init__()
        self.args = args

    def forward(self, predictions, targets):
        # Fourier Transform
        pred_fft = t.fft.rfft(predictions, dim=1)
        target_fft = t.fft.rfft(targets, dim=1)

        # Frequency Loss - Complex
        if self.args.fourier_loss_type == 'complex':
            
            freq_loss = pred_fft - target_fft
            # if self.args.loss == 'MSE':
            #     freq_loss = t.mean(t.abs(freq_loss) ** 2)
            #if self.args.loss == 'MAE':
            freq_loss = t.mean(t.abs(freq_loss))

        # Frequency Loss - Magnitude
        if self.args.fourier_loss_type == 'mag':

            pred_fft_mag = t.abs(pred_fft)
            target_fft_mag = t.abs(target_fft)

            freq_loss = pred_fft_mag - target_fft_mag
            # if self.args.loss == 'MSE': 
            #     freq_loss = t.mean(t.abs(freq_loss) ** 2)
            #if self.args.loss == 'MAE':
            freq_loss = t.mean(t.abs(freq_loss))
        
        # Frequency Loss - Phase
        if self.args.fourier_loss_type == 'phase':

            pred_fft_phase = t.angle(pred_fft)
            target_fft_phase = t.angle(target_fft)

            freq_loss = pred_fft_phase - target_fft_phase
            # if self.args.loss == 'MSE':     
            #     freq_loss = t.mean(t.abs(freq_loss) ** 2)
            #if self.args.loss == 'MAE':
            freq_loss = t.mean(t.abs(freq_loss))

        return freq_loss


# Laguerre Transform
class LaguerreLoss(nn.Module):
    def __init__(self, args):
        super(LaguerreLoss, self).__init__()
        self.args = args
        self.degree = self.args.degree
        self.device = self.args.device

    def forward(self, predictions, targets):
        
        pred_laguerre = laguerre_torch(predictions, degree=self.degree, rtn_data=False, device=self.device)
        target_laguerre = laguerre_torch(targets, degree=self.degree, rtn_data=False, device=self.device)
        
        laguerre_loss = t.mean(t.abs(pred_laguerre - target_laguerre))

        return laguerre_loss


# Legendre Transform
class LegendreLoss(nn.Module):
    def __init__(self, args):
        super(LegendreLoss, self).__init__()
        self.args = args
        self.degree = self.args.degree
        self.device = self.args.device

    def forward(self, predictions, targets):
        
        pred_legendre = legendre_torch(predictions, degree=self.degree, rtn_data=False, device=self.device)
        target_legendre = legendre_torch(targets, degree=self.degree, rtn_data=False, device=self.device)
        
        legendre_loss = t.mean(t.abs(pred_legendre - target_legendre))

        return legendre_loss


# Chebyshev Transform
class ChebyshevLoss(nn.Module):
    def __init__(self, args):
        super(ChebyshevLoss, self).__init__()
        self.args = args
        self.degree = self.args.degree
        self.device = self.args.device

    def forward(self, predictions, targets):

        pred_chebyshev = chebyshev_torch(predictions, degree=self.degree, rtn_data=False, device=self.device)
        target_chebyshev = chebyshev_torch(targets, degree=self.degree, rtn_data=False, device=self.device)
        
        chebyshev_loss = t.mean(t.abs(pred_chebyshev - target_chebyshev))

        return chebyshev_loss


# Hermite Transform
class HermiteLoss(nn.Module):
    def __init__(self, args):
        super(HermiteLoss, self).__init__()
        self.args = args
        self.degree = self.args.degree
        self.device = self.args.device

    def forward(self, predictions, targets):
        
        pred_hermite = hermite_torch(predictions, degree=self.degree, rtn_data=False, device=self.device)
        target_hermite = hermite_torch(targets, degree=self.degree, rtn_data=False, device=self.device)
        
        hermite_loss = t.mean(t.abs(pred_hermite - target_hermite))

        return hermite_loss


# Representation CKA Loss
class ReprCKALoss(nn.Module):
    def __init__(self, args):
        super(ReprCKALoss, self).__init__()
        self.args = args
        self.loss_model = t.load(self.args.loss_model_path)
        self.enc_embedding = self.loss_model.enc_embedding
        self.encoder = self.loss_model.encoder

    def forward(self, predictions, targets):
        # TODO: Implement ReprLoss
        repr_pred = self.enc_embedding(predictions)
        repr_pred = self.encoder(repr_pred)

        repr_target = self.enc_embedding(targets)
        repr_target = self.encoder(repr_target)

        #CKA(Centered Kernel Alignment) loss
        repr_CKA_loss = cka_torch(repr_pred, repr_target)

        return repr_CKA_loss


########################################################################################################
class Loss(nn.Module):
    def __init__(self, args):
        super(Loss, self).__init__()
        self.args = args
        self.alpha = self.args.alpha
        self.base = self.args.base
        self.distance = self.args.distance
        self.temp_to = self.args.temp_to
        self.temp = self.args.temp
        
        if self.args.additional == "window":
            self.additional_loss = WindowShapeLoss(self.base, self.distance, self.temp_to, self.temp)
        elif self.args.additional == "fourier":
            self.additional_loss = FrequencyLoss(self.args)
        elif self.args.additional == "laguerre":
            self.additional_loss = LaguerreLoss(self.args)
        elif self.args.additional == "legendre":
            self.additional_loss = LegendreLoss(self.args)
        elif self.args.additional == "chebyshev":
            self.additional_loss = ChebyshevLoss(self.args)
        elif self.args.additional == "hermite":
            self.additional_loss = HermiteLoss(self.args)
        elif self.args.additional == "repr_cka":
            self.additional_loss = ReprCKALoss(self.args)
        else:
            raise ValueError("Invalid additional loss type")

    def forward(self, predictions, targets):
        if self.args.base == "MSE":
            self.base_loss = nn.MSELoss()
        elif self.args.base == "MAE":
            self.base_loss = nn.L1Loss()
        
        if self.alpha == 0:
            return self.base_loss(predictions, targets)
        else:
            return (1 - self.alpha) * self.base_loss(predictions, targets) + self.alpha * self.additional_loss(predictions, targets)
