import torch
import numpy as np
import torch.optim as optim
import time
import torch.nn as nn
from DNN_BP import *
from data_gen import *

total_start = time.time()

# Choose configuration for channel by commenting or uncommenting
Nt, Nr =  8, 32

Rx, Tx = 0.0, 0.0

# Choose the device to run the code. 
device = torch.device('cuda:0') 

# choose the file you want to save to
file = 'DNN-MS-8x32-6.pt'

# ------------------------------------------------------------------------------------------
# ------------ Below is not for modification. Please don't change anything below------------
# ------------------------------------------------------------------------------------------

Es  = 10                      # mean energy per symbol for 16-QAM
batch = 120                   # number of samples per mini-batch
net = DNN_MS_6(Nt,Nr,device)
# set the optimizer
optimizer = optim.Adam(net.parameters(), lr= 1e-5 )
loss_fn = nn.CrossEntropyLoss()

num = 0
while (num < 10000):
    start = time.time()
    
    snr = torch.randint(low=0, high=6, size=(batch,Nr,1), device=device)*2.0+4.0
    noise_var = noise_variance(snr, Es, Nt)
    n = torch.randn(size=(batch, 2*Nr, 1), device=device)
    n[:,0::2,:] *= ((noise_var*0.5)**0.5)
    n[:,1::2,:] *= ((noise_var*0.5)**0.5)

    y, x, H = data_gen_16QAM_s(batch, Nt, Nr, n, Rx, Tx, device)
    
    y_s, x_s, H_s, n_s = shuffle_data(batch, y, x, H, n)
    
    optimizer.zero_grad()
    P = torch.ones(size=(batch, 2*Nr, 2*Nt, 4), device=device)*0.25
    
    _ , gamma = net(y_s, H_s, P, n_s)

    O = (torch.sigmoid(gamma)).view(-1,4)
    xr = (((x_s+3)/2).view(-1)).type(torch.long)

    loss = loss_fn(O, xr)
    
    if (torch.isnan(loss) == True):
        continue
    
    loss.backward()
    optimizer.step()
    print('------M:{}----N:{}------------{}--------------'.format(Nt,Nr, device))
    print('Cross-Entropy Loss of batch {}: {}'.format(num+1, loss))
    print('Training time for batch     {}: {} seconds'.format(num+1,time.time()-start))
    
    torch.save(net.state_dict(), file)
    print('Succesfully Saved to {}'.format(file))
    
    num += 1
    
print('The total Training Time is {} seconds'.format(time.time()-total_start))

