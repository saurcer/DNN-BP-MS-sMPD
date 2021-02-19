from scipy.linalg import toeplitz
import torch
from scipy.linalg import sqrtm
import numpy as np
from data_gen import *

def noise_variance(SNR, Es, Nt):
    """
    Compute noise_variance given
        1. SNR: Signal-to-Noise Ratio in dB
        2. Es : Mean symbol Energy
        3. Nt : Number of transmitter
    """
    return (Nt*Es)/(10.0**(SNR/10.0))

def correlation_matrix(coef, size):
    """
    Generate the correlation matrix given 
        1. coef: correlation coefficient
        2. size: size of the square matrix
    """
    return coef**(toeplitz(torch.arange(start=0, end=size, step=1)))

def data_gen_16QAM(batch, Nt, Nr, noise_var, Rx, Tx, device):
    """
    Generate data given a scarlar value of noise variance 
    """
    
    # generate symbol signal
    x = torch.randint(low=0, high=4, size=(batch,2*Nt,1), device=device)*2.0-3.0
    
    # generate the channel matrix
    H = torch.empty(size=(batch, 2*Nr, 2*Nt), device=device)
    H[:,0::2,0::2] = H[:,1::2,1::2] = torch.randn(size=(batch, Nr, Nt))*(0.5**0.5) # real part of the matrix
    H[:,0::2,1::2] = H[:,1::2,0::2] = torch.randn(size=(batch, Nr, Nt))*(0.5**0.5) # complex part of the matrix
    H[:,0::2,1::2] *= -1 
    
    # generate noise
    n = torch.randn(size=(batch, 2*Nr, 1), device=device)*((noise_var*0.5)**0.5)
    
    # generate correlation matrix
    Rrx = torch.zeros(size=(2*Nr, 2*Nr), device=device)
    Rtx = torch.zeros(size=(2*Nt, 2*Nt), device=device)
    Rrx[0::2,0::2] = Rrx[1::2,1::2] = torch.as_tensor(sqrtm(correlation_matrix(Rx, Nr)), device=device)
    Rtx[0::2,0::2] = Rtx[1::2,1::2] = torch.as_tensor(sqrtm(correlation_matrix(Tx, Nt)), device=device)
     
    # generate the received signal by y = Hx + n
    corr_H = Rrx@H@Rtx
    y = corr_H@x + n
    
    return y, x, corr_H

def data_gen_16QAM_s(batch, Nt, Nr, n, Rx, Tx, device):
    """
    Generate data given a vector of noise variance 
    """
    # generate symbol signal
    x = torch.randint(low=0, high=4, size=(batch,2*Nt,1), device=device)*2.0-3.0
    
    # generate the channel matrix
    H = torch.empty(size=(batch, 2*Nr, 2*Nt), device=device)
    H[:,0::2,0::2] = H[:,1::2,1::2] = torch.randn(size=(batch, Nr, Nt))*(0.5**0.5)
    H[:,0::2,1::2] = H[:,1::2,0::2] = torch.randn(size=(batch, Nr, Nt))*(0.5**0.5)
    H[:,0::2,1::2] *= -1
    
    # generate correlation matrix
    Rrx = torch.zeros(size=(2*Nr, 2*Nr), device=device)
    Rtx = torch.zeros(size=(2*Nt, 2*Nt), device=device)
    Rrx[0::2,0::2] = Rrx[1::2,1::2] = torch.as_tensor(sqrtm(correlation_matrix(Rx, Nr)), device=device)
    Rtx[0::2,0::2] = Rtx[1::2,1::2] = torch.as_tensor(sqrtm(correlation_matrix(Tx, Nt)), device=device)
     
    # generate the received signal by y = Hx + n
    corr_H = Rrx@H@Rtx
    y = corr_H@x + n
    
    return y, x, corr_H

def shuffle_data(batch, y, x, H, n):
    """
    Shuffle the data for training
    """
    shuffle_ind = torch.randperm(batch)
    return y[shuffle_ind], x[shuffle_ind], H[shuffle_ind], n[shuffle_ind]