import torch
import numpy as np
import torch.optim as optim
import time
import torch.nn as nn
from DNN_BP import *
from data_gen import *

total_start = time.time()

# Choose configuration for channel by commenting or uncommenting
Nt, Nr, L =  8, 32, 5

Rx, Tx = 0.0, 0.0

# Choose the device to run the code. 
device = torch.device('cuda:0') 

# choose the file you want to save to
file = 'DNN-dBP-8x32-10.pt'

# SNR = [4,6,8,10,12,14]        # Signal-to-Noise ratio in dB
Es  = 10                      # mean energy per symbol for 16-QAM
batch = 120                   # number of samples per batch
# noise_var = [noise_variance(snr, Es, Nt) for snr in SNR]     # compute the noise variance for each SNR
net = DNN_dBP_10(Nt,Nr,device)
# set the optimizer
optimizer = optim.Adam(net.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

num = 0
while (num < 1000):
    start = time.time()

    # generate the noise vector
    snr = torch.randint(low=0, high=6, size=(batch,Nr,1), device=device)*2.0+4.0
    noise_var = noise_variance(snr, Es, Nt)
    n = torch.empty(size=(batch,2*Nr,1), device=device)
    n[:,0::2,:] = n[:,1::2,:] = ((noise_var*0.5)**0.5)

    # generate the data given noise vector
    y, x, H = data_gen_16QAM_s(batch, Nt, Nr, n, Rx, Tx, device)

    # shuffle the data
    y_s, x_s, H_s, n_s = shuffle_data(batch, y, x, H, n)
    
    optimizer.zero_grad()
    P = torch.ones(size=(batch, 2*Nr, 2*Nt, 4), device=device)*0.25
    
    _ , gamma = net(y_s, H_s, P, n_s)

    O = (torch.sigmoid(gamma)).view(-1,4)
    xr = (((x_s+3)/2).view(-1)).type(torch.long)

    loss = loss_fn(O, xr)
    
    loss.backward()
    optimizer.step()
    print('------M:{}----N:{}-----L:{}----------{}--------------'.format(Nt,Nr,L, device))
    print('Cross-Entropy Loss of batch {}: {}'.format(num+1, loss))
    print('Training time for batch     {}: {} seconds'.format(num+1,time.time()-start))
    
    torch.save(net.state_dict(), file)
    print('Succesfully Saved to {}'.format(file))
    
    num += 1
    
print('The total Training Time is {} seconds'.format(time.time()-total_start))