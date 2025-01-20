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
from torch.jit import script
import numpy as np
import pdb

from utils.polynomials import laguerre_torch, hermite_torch, legendre_torch, chebyshev_torch
from utils.chronos_repr import ChronosRepr
from utils.adversarial import AdversarialLoss
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


# Quantile Loss
class QuantileLoss(nn.Module):
    def __init__(self, args):
        super(QuantileLoss, self).__init__()
        self.args = args
        self.q = self.args.q

    def forward(self, inputs, predictions, targets):
        diff = predictions - targets
        loss = t.max(self.q * diff, (self.q - 1) * diff)
        return t.mean(loss) 


# Representation Loss
class ReprLoss(nn.Module):
    def __init__(self, args):
        super(ReprLoss, self).__init__()
        self.args = args
        self.device = self.args.device
        self.base_loss = nn.MSELoss()
        self.alpha = self.args.alpha
        
    def forward(self, inputs, predictions, targets):
        batch_size, seq_len, num_features = inputs.shape
        seq_len_pred_tar = predictions.shape[1]

        base_loss = self.base_loss(predictions, targets)

        inputs = inputs.permute(0, 2, 1).reshape(-1, seq_len).to(self.device) # (batch_size * num_features, seq_len)
        predictions = predictions.permute(0, 2, 1).reshape(-1, seq_len_pred_tar).to(self.device) # (batch_size * num_features, seq_len)
        targets = targets.permute(0, 2, 1).reshape(-1, seq_len_pred_tar).to(self.device) # (batch_size * num_features, seq_len)

        repr_model = ChronosRepr(self.args.repr_model)

        # Process sequences one by one for compatibility with chronos_repr
        repr_inp = t.stack([
            repr_model.get_repr(inputs[i].to(self.device))[0,-1,:] # last special token
            for i in range(inputs.shape[0])
        ])
        repr_pred = t.stack([
            repr_model.get_repr(predictions[i].to(self.device))[0,-1,:] # last special token
            for i in range(predictions.shape[0])
        ])
        repr_tar = t.stack([
            repr_model.get_repr(targets[i].to(self.device))[0,-1,:] # last special token
            for i in range(targets.shape[0])
        ])
        
        hidden_dim = 512  # Fixed hidden dimension
        repr_inp = repr_inp.view(batch_size, num_features, hidden_dim)
        repr_pred = repr_pred.view(batch_size, num_features, hidden_dim)
        repr_tar = repr_tar.view(batch_size, num_features, hidden_dim)

        # Compute loss per sample and per variable
        abs_loss = t.mean(t.abs(repr_pred - repr_tar), dim=-1)  # [batch_size, num_features]
        # Convert cosine similarity to a loss (1 - cosine similarity)
        # Higher loss if the vectors are not aligned (cos_sim < 1)
        one_tensor = t.tensor(1.0, device=repr_pred.device, requires_grad=True)
        cos_loss = one_tensor - nn.functional.cosine_similarity(repr_pred, repr_tar, dim=-1)  # [batch_size, num_features]
        
        # Average over variables and batch
        repr_loss = t.mean(abs_loss + cos_loss)  # Final scalar loss

        return (1 - self.alpha) * base_loss + self.alpha * repr_loss

# Learnable Network with Adversarial Loss
class LearnableNetwork(nn.Module):
    def __init__(self, args):
        super(LearnableNetwork, self).__init__()
        self.args = args
        self.device = self.args.device

        # # Learnable Network
        # self.repr_network = nn.Sequential(
        #     nn.Linear(args.enc_in, 100),
        #     nn.ReLU(),
        #     nn.Linear(100, 100),
        #     nn.ReLU(),
        #     nn.Linear(100, 64)
        # ).to(self.device)

        # Discriminator
        self.discriminator = nn.Sequential(
            nn.Linear(512, 1000),
            nn.ReLU(),
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, 250),
            nn.ReLU(),
            nn.Linear(250, 1),
            nn.Sigmoid()
        ).to(self.device)

    def forward(self, inputs, predictions, targets):
        X_Y_hat = t.cat((inputs, predictions), dim=1).to(self.device) # (batch_size, seq_len + seq_len_pred, num_features(enc_in))
        X_Y = t.cat((inputs, targets), dim=1).to(self.device) # (batch_size, seq_len + seq_len_pred, num_features(enc_in))

        # repr_X_Y_hat = self.repr_network(X_Y_hat)
        # repr_X_Y = self.repr_network(X_Y)

        batch_size, X_Y_len, num_features = X_Y.shape

        X_Y_hat = X_Y_hat.permute(0, 2, 1).reshape(-1, X_Y_len).to(self.device) # (batch_size * num_features, seq_len + seq_len_pred)
        X_Y = X_Y.permute(0, 2, 1).reshape(-1, X_Y_len).to(self.device) # (batch_size * num_features, seq_len + seq_len_pred)

        repr_model = ChronosRepr(self.args.repr_model)
        
        repr_X_Y_hat = t.stack([
            repr_model.get_repr(X_Y_hat[i].to(self.device))[0,-1,:] # last special token
            for i in range(X_Y_hat.shape[0])
        ])
        repr_X_Y = t.stack([
            repr_model.get_repr(X_Y[i].to(self.device))[0,-1,:] # last special token
            for i in range(X_Y.shape[0])
        ])

        hidden_dim = 512  # Fixed hidden dimension
        repr_X_Y_hat = repr_X_Y_hat.view(-1, hidden_dim).float()
        repr_X_Y = repr_X_Y.view(-1, hidden_dim).float()

        loss_F = 0
        loss_D = 0
        for i in range(batch_size * num_features):
            # Discriminator outputs
            pred = self.discriminator(repr_X_Y_hat[i].to(self.device))  # For predictions (fake)
            pred_detached = pred.detach()
            true = self.discriminator(repr_X_Y[i].to(self.device))      # For targets (real)

            adversarial_loss = AdversarialLoss()
            loss_F += adversarial_loss.generator_loss(pred).to(self.device)
            loss_D += adversarial_loss.discriminator_loss(pred_detached, true).to(self.device)

        return loss_F, loss_D

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
        elif self.args.additional == "quantile":
            self.additional_loss = QuantileLoss(self.args)
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
            base_loss = self.base_loss(predictions, targets)
            additional_loss = self.additional_loss(predictions, targets)
            #print("base_loss: ", base_loss)
            #print("additional_loss: ", additional_loss)
            return (1 - self.alpha) * base_loss + self.alpha * additional_loss
