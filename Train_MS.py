import torch
import numpy as np
import torch.optim as optim
import time
import torch.nn as nn
from DNN_BP import *
from data_gen import *

total_start = time.time()

# Choose configuration for channel by commenting or uncommenting
Nt, Nr, L =  8, 32, 10

Rx, Tx = 0.0, 0.0

# Choose the device to run the code. 
device = torch.device('cuda:0') 

# choose the file you want to save to
file = 'DNN-MS-8x32-10.pt'

# ------------------------------------------------------------------------------------------
# ------------ Below is not for modification. Please don't change anything below------------
# ------------------------------------------------------------------------------------------

SNR = [4,6,8,10,12,14]        # Signal-to-Noise ratio in dB
Es  = 10                      # mean energy per symbol for 16-QAM
batch = 120                   # number of samples per mini-batch
noise_var = [noise_variance(snr, Es, Nt) for snr in SNR]     # compute the noise variance for each SNR
net = DNN_MS_10(Nt,Nr,device)
# set the optimizer
optimizer = optim.Adam(net.parameters(), lr=1e-5)

# placeholder for data generation
x = torch.zeros((batch, 2*Nt, 1), device=device)
y = torch.zeros((batch, 2*Nr, 1), device=device)
H = torch.zeros((batch, 2*Nr, 2*Nt), device=device)

loss_fn = nn.CrossEntropyLoss()

num = 0
while (num < 10000):
    start = time.time()
    
    y[  0: 20], x[  0: 20], H[  0: 20] = data_gen_16QAM(20, Nt, Nr, noise_var[0], Rx, Tx, device)
    y[ 20: 40], x[ 20: 40], H[ 20: 40] = data_gen_16QAM(20, Nt, Nr, noise_var[1], Rx, Tx, device)
    y[ 40: 60], x[ 40: 60], H[ 40: 60] = data_gen_16QAM(20, Nt, Nr, noise_var[2], Rx, Tx, device)
    y[ 60: 80], x[ 60: 80], H[ 60: 80] = data_gen_16QAM(20, Nt, Nr, noise_var[3], Rx, Tx, device)
    y[ 80:100], x[ 80:100], H[ 80:100] = data_gen_16QAM(20, Nt, Nr, noise_var[4], Rx, Tx, device)
    y[100:120], x[100:120], H[100:120] = data_gen_16QAM(20, Nt, Nr, noise_var[5], Rx, Tx, device)
    
    optimizer.zero_grad()
    P = torch.ones(size=(20, 2*Nr, 2*Nt, 4), device=device)*0.25
    
    _ , gamma_0 = net(y[  0: 20], H[  0: 20], P, noise_var[0])
    _ , gamma_1 = net(y[ 20: 40], H[ 20: 40], P, noise_var[1])
    _ , gamma_2 = net(y[ 40: 60], H[ 40: 60], P, noise_var[2])
    _ , gamma_3 = net(y[ 60: 80], H[ 60: 80], P, noise_var[3])
    _ , gamma_4 = net(y[ 80:100], H[ 80:100], P, noise_var[4])
    _ , gamma_5 = net(y[100:120], H[100:120], P, noise_var[5])

    O = (torch.sigmoid(torch.cat((gamma_0, gamma_1, gamma_2, gamma_3, gamma_4, gamma_5), 0))).view(-1,4)
    xr = (((x+3)/2).view(-1)).type(torch.long)

    loss = loss_fn(O, xr)
    
    if (torch.isnan(loss) == True):
        continue
    
    loss.backward()
    optimizer.step()
    print('------M:{}----N:{}-----L:{}----------{}--------------'.format(Nt,Nr,L, device))
    print('Cross-Entropy Loss of batch {}: {}'.format(num+1, loss))
    print('Training time for batch     {}: {} seconds'.format(num+1,time.time()-start))
    
    torch.save(net.state_dict(), file)
    print('Succesfully Saved to {}'.format(file))
    
    num += 1
    
print('The total Training Time is {} seconds'.format(time.time()-total_start))