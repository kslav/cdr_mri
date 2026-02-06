#!/usr/bin/env python

import numpy as np
import torch

'''
Defines complex-valued arithmetic using PyTorch's native complex number support.
Also provides conversion utilities between numpy complex and torch complex.
'''

def c2r(z):
    ''' Convert from complex numpy array to 2-channel real numpy array '''
    assert type(z) is np.ndarray, 'Must be numpy.ndarray'
    return np.stack((z.real, z.imag), axis=-1)

def r2c(x):
    ''' Convert from 2-channel real numpy array to complex numpy array '''
    assert type(x) is np.ndarray, 'Must be numpy.ndarray'
    return x[...,0] + 1j * x[...,1]

def np_to_torch_complex(z):
    ''' Convert numpy complex array to torch complex tensor '''
    assert np.iscomplexobj(z), 'Input must be complex-valued numpy array'
    return torch.view_as_complex(torch.from_numpy(np.stack((z.real, z.imag), axis=-1)).contiguous())

def torch_to_np_complex(z):
    ''' Convert torch complex tensor to numpy complex array '''
    assert torch.is_complex(z), 'Input must be complex-valued torch tensor'
    z_np = torch.view_as_real(z).cpu().numpy()
    return z_np[..., 0] + 1j * z_np[..., 1]

def real2ch_to_complex(x):
    ''' Convert 2-channel real tensor (..., 2) to native complex tensor '''
    if isinstance(x, torch.Tensor):
        # Ensure contiguous for view_as_complex
        if x.shape[-1] != 2:
            raise ValueError(f"Expected last dimension to be 2 (real, imag), got shape {x.shape}")
        return torch.view_as_complex(x.contiguous())
    else:
        raise TypeError('Input must be torch.Tensor')

def complex_to_real2ch(z):
    ''' Convert native complex tensor to 2-channel real tensor (..., 2) '''
    if isinstance(z, torch.Tensor):
        if not torch.is_complex(z):
            raise ValueError('Input must be complex-valued tensor')
        return torch.view_as_real(z)
    else:
        raise TypeError('Input must be torch.Tensor')

def zmul(x1, x2):
    ''' Complex multiplication - works with both native complex and 2-channel real '''
    # Check if inputs are native complex tensors
    if isinstance(x1, torch.Tensor) and torch.is_complex(x1):
        return x1 * x2
    elif isinstance(x1, np.ndarray) and np.iscomplexobj(x1):
        return x1 * x2

    # Legacy 2-channel real format
    xr = x1[...,0] * x2[...,0] - x1[...,1] * x2[...,1]
    xi = x1[...,0] * x2[...,1] + x1[...,1] * x2[...,0]
    if type(x1) is np.ndarray:
        return np.stack((xr, xi), axis=-1)
    elif type(x1) is torch.Tensor:
        return torch.stack((xr, xi), dim=-1)
    else:
        return xr, xi

def zconj(x):
    ''' Complex conjugate - works with both native complex and 2-channel real '''
    # Check if input is native complex tensor
    if isinstance(x, torch.Tensor) and torch.is_complex(x):
        return torch.conj(x)
    elif isinstance(x, np.ndarray) and np.iscomplexobj(x):
        return np.conj(x)

    # Legacy 2-channel real format
    if type(x) is np.ndarray:
        return np.stack((x[...,0], -x[...,1]), axis=-1)
    elif type(x) is torch.Tensor:
        return torch.stack((x[...,0], -x[...,1]), dim=-1)
    else:
        return x[...,0], -x[...,1]

def zabs(x):
    ''' Complex magnitude - works with both native complex and 2-channel real '''
    # Check if input is native complex tensor
    if isinstance(x, torch.Tensor) and torch.is_complex(x):
        return torch.abs(x)
    elif isinstance(x, np.ndarray) and np.iscomplexobj(x):
        return np.abs(x)

    # Legacy 2-channel real format
    if type(x) is np.ndarray:
        return np.sqrt(zmul(x, zconj(x)))[...,0]
    elif type(x) is torch.Tensor:
        return torch.sqrt(zmul(x, zconj(x)))[...,0]
    else:
        return -1.
