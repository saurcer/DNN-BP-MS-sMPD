import torch
import torch.nn as nn
import torch.nn.functional as F

def cross_entropy_loss(x, output, device):
    batch = x.shape[0]
    M = x.shape[1]
    s = torch.tensor([-3,-1,1,3], device = device)
    x_onehot = (torch.eq(s,x)).type(torch.float)
    out = torch.sum(torch.log(torch.squeeze(output))*x_onehot)
    return -out/(batch*2*M)

class DBP_layer(nn.Module):
    def __init__(self, M, N, device):
        super().__init__()
        self.M, self.N = M, N   # number of transmitters and receivers respectively
        damp = torch.ones((1), device=device)*0.5 
        self.damp = nn.Parameter(damp)
        self.device = device
        
    def forward(self, y, H, P, noise_var):
        s = torch.tensor([-3.0, -1.0, 1.0, 3.0],device=self.device)
        mean_z = torch.sum(H*(P@s), axis=2, keepdim = True) - H*(P@s)
        var_z = torch.sum((H**2)*(P@(s*s)-(P@s)**2),axis=2,keepdim=True)-(H**2)*(P@(s*s)-(P@s)**2)+noise_var
        beta = ((torch.unsqueeze(2*H*(y - mean_z), dim=3))@(s-s[0]).view(1,s.shape[0])-(torch.unsqueeze(H**2, dim=3))@(s**2-(s[0])**2).view(1,s.shape[0]))/(2.0*torch.unsqueeze(var_z, dim=3))
        gamma = torch.sum(beta, dim=1,keepdim=True)
        alpha = gamma-beta
        Post_P = F.softmax(alpha, dim=3)
        P1 = (1-self.damp)*Post_P + self.damp*P.clone().detach()
        return P1, gamma

class DNN_dBP_5(nn.Module):
    def __init__(self, M, N, device):
        super().__init__()
        self.dbp_1 = DBP_layer(M, N, device)
        self.dbp_2 = DBP_layer(M, N, device)
        self.dbp_3 = DBP_layer(M, N, device)
        self.dbp_4 = DBP_layer(M, N, device)
        self.dbp_5 = DBP_layer(M, N, device)
        
    def forward(self, y, H, P, noise_var):
        P, _     = self.dbp_1(y, H, P, noise_var)
        P, _     = self.dbp_2(y, H, P, noise_var)
        P, _     = self.dbp_3(y, H, P, noise_var)
        P, _     = self.dbp_4(y, H, P, noise_var)
        P, gamma = self.dbp_5(y, H, P, noise_var)
        return P, gamma
    
class DNN_dBP_10(nn.Module):
    def __init__(self, M, N, device):
        super().__init__()
        self.dbp_1 = DBP_layer(M, N, device)
        self.dbp_2 = DBP_layer(M, N, device)
        self.dbp_3 = DBP_layer(M, N, device)
        self.dbp_4 = DBP_layer(M, N, device)
        self.dbp_5 = DBP_layer(M, N, device)
        self.dbp_6 = DBP_layer(M, N, device)
        self.dbp_7 = DBP_layer(M, N, device)
        self.dbp_8 = DBP_layer(M, N, device)
        self.dbp_9 = DBP_layer(M, N, device)
        self.dbp_10 = DBP_layer(M, N, device)
        
    def forward(self, y, H, P, noise_var):
        P, _     = self.dbp_1(y, H, P, noise_var)
        P, _     = self.dbp_2(y, H, P, noise_var)
        P, _     = self.dbp_3(y, H, P, noise_var)
        P, _     = self.dbp_4(y, H, P, noise_var)
        P, _     = self.dbp_5(y, H, P, noise_var)
        P, _     = self.dbp_6(y, H, P, noise_var)
        P, _     = self.dbp_7(y, H, P, noise_var)
        P, _     = self.dbp_8(y, H, P, noise_var)
        P, _     = self.dbp_9(y, H, P, noise_var)
        P, gamma = self.dbp_10(y, H, P, noise_var)
        return P, gamma
      
class MS_layer(nn.Module):
    def __init__(self, M, N, device):
        super().__init__()
        self.M, self.N = M, N   # number of transmitters and receivers respectively
        damp = torch.ones((1,1,1,1), device=device)*0.5
        lamb = torch.ones((1,2*N, 2*M, 1), device=device)*0.5
        self.damp = nn.Parameter(damp)
        self.lamb = nn.Parameter(lamb)
        self.device = device
        
    def forward(self, y, H, Pre_P, noise_var):
        P = F.normalize(Pre_P, p=1, dim=3)
        s = torch.tensor([-3.0, -1.0, 1.0, 3.0],device=self.device)
        mean_z = torch.sum(H*(P@s), axis=2, keepdim = True) - H*(P@s)
        var_z = torch.sum((H**2)*(P@(s*s)-(P@s)**2),axis=2,keepdim=True)-(H**2)*(P@(s*s)-(P@s)**2)+noise_var
        beta = ((torch.unsqueeze(2*H*(y - mean_z), dim=3))@(s-s[0]).view(1,s.shape[0])-(torch.unsqueeze(H**2, dim=3))@(s**2-(s[0])**2).view(1,s.shape[0]))/(2.0*torch.unsqueeze(var_z, dim=3))
        gamma = torch.sum(beta, dim=1,keepdim=True)
        alpha = gamma - beta
        Post_P = torch.exp(alpha - (torch.amax(alpha, dim=3, keepdim=True)))
        P1 = (1-self.damp)*self.lamb*Post_P + self.damp*P.clone().detach()
        return P1, gamma
    
class DNN_MS_6(nn.Module):
    def __init__(self, M, N, device):
        super().__init__()
        self.dbp_1 = MS_layer(M, N, device)
        self.dbp_2 = MS_layer(M, N, device)
        self.dbp_3 = MS_layer(M, N, device)
        self.dbp_4 = MS_layer(M, N, device)
#         self.dbp_5 = MS_layer(M, N, device)
        self.dbp_6 = MS_layer(M, N, device)
        
    def forward(self, y, H, P, noise_var):
        P, _     = self.dbp_1(y, H, P, noise_var)
        P, _     = self.dbp_2(y, H, P, noise_var)
        P, _     = self.dbp_3(y, H, P, noise_var)
        P, _     = self.dbp_4(y, H, P, noise_var)
#         P, _     = self.dbp_5(y, H, P, noise_var)
        P, gamma = self.dbp_6(y, H, P, noise_var)
        return P, gamma

class DNN_MS_10(nn.Module):
    def __init__(self, M, N, device):
        super().__init__()
        self.dbp_1 = MS_layer(M, N, device)
        self.dbp_2 = MS_layer(M, N, device)
        self.dbp_3 = MS_layer(M, N, device)
        self.dbp_4 = MS_layer(M, N, device)
        self.dbp_5 = MS_layer(M, N, device)
        self.dbp_6 = MS_layer(M, N, device)
        self.dbp_7 = MS_layer(M, N, device)
        self.dbp_8 = MS_layer(M, N, device)
        self.dbp_9 = MS_layer(M, N, device)
#         self.dbp_10 = MS_layer(M, N, device)
        self.dbp_11 = MS_layer(M, N, device)
        
    def forward(self, y, H, P, noise_var):
        P, _     = self.dbp_1(y, H, P, noise_var)
        P, _     = self.dbp_2(y, H, P, noise_var)
        P, _     = self.dbp_3(y, H, P, noise_var)
        P, _     = self.dbp_4(y, H, P, noise_var)
        P, _     = self.dbp_5(y, H, P, noise_var)
        P, _     = self.dbp_6(y, H, P, noise_var)
        
        P, _     = self.dbp_7(y, H, P, noise_var)
        P, _     = self.dbp_8(y, H, P, noise_var)
        P, _     = self.dbp_9(y, H, P, noise_var)
#         P, _     = self.dbp_10(y, H, P, noise_var)
        P, gamma = self.dbp_11(y, H, P, noise_var)
        return P, gamma
    
    
class sMPD_layer(nn.Module):
    def __init__(self, M, device):
        super().__init__()
        self.M = M  # number of transmitters and receivers respectively
        damp = torch.ones((1), dtype=torch.float, device=device)*0.5 
        self.damp = nn.Parameter(damp)
        w = torch.ones((1,2*M, 1), dtype=torch.float, device=device)*0.5
        self.w = nn.Parameter(w)
        b = torch.ones((1,2*M, 1), dtype=torch.float, device=device)*0.5 
        self.b = nn.Parameter(b)
        self.device = device
        
    def forward(self, z, J, Pre_P, var_j, device):
        P = F.normalize(Pre_P, p=1, dim=2)
        s = torch.tensor([[-3.0, -1.0, 1.0, 3.0]], device=self.device).T
        J_diag = torch.unsqueeze(torch.diagonal(J, offset=0, dim1=1, dim2=2), dim=2)
        mean_v = torch.sum(J*torch.transpose(P@s,1,2),dim=2,keepdim=True) - J_diag*(P@s) 
        L = (2*(z-mean_v)-J_diag*(s+s[0]).T)*(J_diag*(s-s[0]).T)/(2.0*var_j)
        L1 = L - torch.max(L,dim=2,keepdim=True).values
        P1 = self.w*torch.exp(L1)-self.b
        P2 = (1-self.damp)*P1 + self.damp*P.clone().detach()
        return P2, L1
    
class DNN_sMPD_6(nn.Module):
    def __init__(self, M, device):
        super().__init__()
        self.sMPD_1 = sMPD_layer(M, device)
        self.sMPD_2 = sMPD_layer(M, device)
        self.sMPD_3 = sMPD_layer(M, device)
        self.sMPD_4 = sMPD_layer(M, device)
        self.sMPD_5 = sMPD_layer(M, device)
        self.sMPD_6 = sMPD_layer(M, device)
        
    def forward(self, z, J, P, var_j, device):
        P, _     = self.sMPD_1(z, J, P, var_j, device)
        P, _     = self.sMPD_2(z, J, P, var_j, device)
        P, _     = self.sMPD_3(z, J, P, var_j, device)
        P, _     = self.sMPD_4(z, J, P, var_j, device)
        P, _     = self.sMPD_5(z, J, P, var_j, device)
        P, L     = self.sMPD_6(z, J, P, var_j, device)
        return P, L

class DNN_sMPD_3(nn.Module):
    def __init__(self, M, device):
        super().__init__()
        self.sMPD_1 = sMPD_layer(M, device)
        self.sMPD_2 = sMPD_layer(M, device)
        self.sMPD_3 = sMPD_layer(M, device)
        
    def forward(self, z, J, P, var_j, device):
        P, _ = self.sMPD_1(z, J, P, var_j, device)
        P, _ = self.sMPD_2(z, J, P, var_j, device)
        P, L = self.sMPD_3(z, J, P, var_j, device)
        return P, L

def sMPD_transform(y, H, P, noise, Nr, device):
    J = ((torch.transpose(H, 1, 2))@H)/(2*Nr)
    z = ((torch.transpose(H, 1, 2))@y)/(2*Nr)
    s = torch.tensor([[-3.0, -1.0, 1.0, 3.0]], device=device).T
    J_diag = torch.unsqueeze(torch.diagonal(J, offset=0, dim1=1, dim2=2), dim=2)
    var_v = torch.sum((J**2)*torch.transpose(P@(s*s)-(P@s)**2, 1, 2),dim=2,keepdim=True)-(J_diag**2)*(P@(s*s)-(P@s)**2)
    V_max = torch.max(var_v, dim=1, keepdim=True).values
    J_max = torch.max(torch.max(J, dim=1, keepdim=True).values, dim=2, keepdim=True).values

    var_j = (J_max**2)*V_max + noise/(2*Nr)
    return z, J, var_j