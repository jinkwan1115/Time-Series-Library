import math
import numpy as np
import torch

from numpy.polynomial import Chebyshev
from numpy.polynomial import Hermite
from numpy.polynomial import Laguerre
from numpy.polynomial import Legendre

def laguerre_torch(data, degree, rtn_data=False, device='cpu'):
    degree += 1
    ndim = data.ndim
    shape = data.shape

    if ndim == 2: # [T, F]
        B = 1
        T = shape[0] # [1, T, F]
    elif ndim == 3: # [B, T, F]
        B,T = shape[:2] 
        data = data.permute(1, 0, 2).reshape(T, -1) # [T, B*F]
    else:
        raise ValueError('The input data should be 1D or 2D.')
    
    tvals = np.linspace(0, 5, T)
    laguerre_polys = np.array([Laguerre.basis(i)(tvals).astype(np.float32) for i in range(degree)])
    laguerre_polys = torch.tensor(laguerre_polys, dtype=torch.float32, device=device)
    # [degree, T]

    data = data.to(device)
    coeffs_candidate = torch.mm(laguerre_polys, data) / T
    coeffs = coeffs_candidate.transpose(0, 1)
    # [B*F, degree]

    if rtn_data:
        reconstructed_data = torch.mm(coeffs, laguerre_polys)
        reconstructed_data = reconstructed_data.reshape(
            B, -1, T).permute(0, 2, 1)

        if ndim == 2:
            reconstructed_data = reconstructed_data.squeeze(0)
        return coeffs, reconstructed_data
    else:
        return coeffs

def hermite_torch(data, degree, rtn_data=False, device='cpu'):
    degree += 1
    ndim = data.ndim
    shape = data.shape

    if ndim == 2: # [T, F]
        B = 1
        T = shape[0] # [1, T, F]
    elif ndim == 3: # [B, T, F]
        B,T = shape[:2] 
        data = data.permute(1, 0, 2).reshape(T, -1) # [T, B*F]
    else:
        raise ValueError('The input data should be 1D or 2D.')
    
    tvals = np.linspace(-5, 5, T)
    hermite_polys = np.array([Hermite.basis(i)(tvals).astype(np.float32) for i in range(degree)])
    hermite_polys = torch.tensor(hermite_polys, dtype=torch.float32, device=device)
    # [degree, T]

    data = data.to(device)
    coeffs_candidate = torch.mm(hermite_polys, data) / T
    coeffs = coeffs_candidate.transpose(0, 1)
    # [B*F, degree]

    if rtn_data:
        reconstructed_data = torch.mm(coeffs, hermite_polys)
        reconstructed_data = reconstructed_data.reshape(
            B, -1, T).permute(0, 2, 1)

        if ndim == 2:
            reconstructed_data = reconstructed_data.squeeze(0)
        return coeffs, reconstructed_data
    else:
        return coeffs

def legendre_torch(data, degree, rtn_data=False, device='cpu'):
    degree += 1
    ndim = data.ndim
    shape = data.shape

    if ndim == 2: # [T, F]
        B = 1
        T = shape[0] # [1, T, F]
    elif ndim == 3: # [B, T, F]
        B,T = shape[:2] 
        data = data.permute(1, 0, 2).reshape(T, -1) # [T, B*F]
    else:
        raise ValueError('The input data should be 1D or 2D.')

    tvals = np.linspace(-1, 1, T) # The Legendre series are defined in[-1, 1]
    legendre_polys = np.array([Legendre.basis(i)(tvals).astype(np.float32) for i in range(degree)])
    legendre_polys = torch.tensor(legendre_polys, dtype=torch.float32, device=device)
    # [degree, T]

    data = data.to(device)
    coeffs_candidate = torch.mm(legendre_polys, data) / T
    coeffs = coeffs_candidate.transpose(0, 1)
    # [B*F, degree]

    if rtn_data:
        reconstructed_data = torch.mm(coeffs, legendre_polys)
        reconstructed_data = reconstructed_data.reshape(
            B, -1, T).permute(0, 2, 1)

        if ndim == 2:
            reconstructed_data = reconstructed_data.squeeze(0)
        return coeffs, reconstructed_data
    else:
        return coeffs

def chebyshev_torch(data, degree, rtn_data=False, device='cpu'):
    degree += 1
    ndim = data.ndim
    shape = data.shape

    if ndim == 2: # [T, F]
        B = 1
        T = shape[0] # [1, T, F]
    elif ndim == 3: # [B, T, F]
        B,T = shape[:2] 
        data = data.permute(1, 0, 2).reshape(T, -1) # [T, B*F]
    else:
        raise ValueError('The input data should be 1D or 2D.')

    tvals = np.linspace(-1, 1, T) # The Legendre series are defined in[-1, 1]
    chebyshev_polys = np.array([Chebyshev.basis(i)(tvals).astype(np.float32) for i in range(degree)])
    chebyshev_polys = torch.tensor(chebyshev_polys, dtype=torch.float32, device=device)
    # [degree, T]

    data = data.to(device)
    coeffs_candidate = torch.mm(chebyshev_polys, data) / T
    coeffs = coeffs_candidate.transpose(0, 1)
    # [B*F, degree]

    if rtn_data:
        reconstructed_data = torch.mm(coeffs, chebyshev_polys)
        reconstructed_data = reconstructed_data.reshape(
            B, -1, T).permute(0, 2, 1)

        if ndim == 2:
            reconstructed_data = reconstructed_data.squeeze(0)
        return coeffs, reconstructed_data
    else:
        return coeffs