import torch
import numpy as np
import torch.optim as optim
import time
import torch.nn as nn
from DNN_BP import *
from data_gen import *

total_start = time.time()

# Choose configuration for channel by commenting or uncommenting
Nt, Nr, L =  8, 32, 3

Rx, Tx = 0.0, 0.0

# Choose the device to run the code. 
device = torch.device('cuda:0') 

# choose the file you want to save to
file = 'DNN-sMPD-3.pt'

# ------------------------------------------------------------------------------------------
# ------------ Below is not for modification. Please don't change anything below------------
# ------------------------------------------------------------------------------------------

SNR = [4,6,8,10,12,14]        # Signal-to-Noise ratio in dB
Es  = 10                      # mean energy per symbol for 16-QAM
batch = 120                   # number of samples per mini-batch
noise_var = [noise_variance(snr, Es, Nt) for snr in SNR]     # compute the noise variance for each SNR
net = DNN_sMPD_3(Nt,device)
# set the optimizer
optimizer = optim.Adam(net.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# placeholder for data generation
x = torch.zeros((batch, 2*Nt, 1), device=device)
P = torch.ones(size=(20, 2*Nt, 4), device=device)*0.25

num = 0
while (num < 1000):
    start = time.time()
    
    y, x[  0: 20], H = data_gen_16QAM(20, Nt, Nr, noise_var[0], Rx, Tx, device)
    z_0, J_0, var_j_0 = sMPD_transform(y, H, P, noise_var[0], Nr, device)
    y, x[ 20: 40], H = data_gen_16QAM(20, Nt, Nr, noise_var[1], Rx, Tx, device)
    z_1, J_1, var_j_1 = sMPD_transform(y, H, P, noise_var[1], Nr, device)
    y, x[ 40: 60], H = data_gen_16QAM(20, Nt, Nr, noise_var[2], Rx, Tx, device)
    z_2, J_2, var_j_2 = sMPD_transform(y, H, P, noise_var[2], Nr, device)
    y, x[ 60: 80], H = data_gen_16QAM(20, Nt, Nr, noise_var[3], Rx, Tx, device)
    z_3, J_3, var_j_3 = sMPD_transform(y, H, P, noise_var[3], Nr, device)
    y, x[ 80:100], H = data_gen_16QAM(20, Nt, Nr, noise_var[4], Rx, Tx, device)
    z_4, J_4, var_j_4 = sMPD_transform(y, H, P, noise_var[4], Nr, device)
    y, x[100:120], H = data_gen_16QAM(20, Nt, Nr, noise_var[5], Rx, Tx, device)
    z_5, J_5, var_j_5 = sMPD_transform(y, H, P, noise_var[5], Nr, device)
    
    optimizer.zero_grad()
    
#     P = torch.ones(size=(20, 2*Nt, 4), device=device)*0.25
    P_0 , L_0 = net(z_0, J_0, P, var_j_0, device)
#     P = torch.ones(size=(20, 2*Nt, 4), device=device)*0.25
    P_1 , L_1 = net(z_1, J_1, P, var_j_1, device)
#     P = torch.ones(size=(20, 2*Nt, 4), device=device)*0.25
    P_2 , L_2 = net(z_2, J_2, P, var_j_2, device)
#     P = torch.ones(size=(20, 2*Nt, 4), device=device)*0.25
    P_3 , L_3 = net(z_3, J_3, P, var_j_3, device)
#     P = torch.ones(size=(20, 2*Nt, 4), device=device)*0.25
    P_4 , L_4 = net(z_4, J_4, P, var_j_4, device)
#     P = torch.ones(size=(20, 2*Nt, 4), device=device)*0.25
    P_5 , L_5 = net(z_5, J_5, P, var_j_5, device)

    O = (torch.sigmoid(torch.cat((L_0, L_1, L_2, L_3, L_4, L_5), 0))).view(-1,4)
    xr = (((x+3)/2).view(-1)).type(torch.long)

#     loss = cross_entropy_loss(x, O, device)
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