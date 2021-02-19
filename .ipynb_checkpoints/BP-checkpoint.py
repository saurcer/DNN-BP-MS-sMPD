import torch
import torch.nn.functional as F

def BP_16QAM_layer( y, H, noise_var, P, delta, device):
    s = torch.tensor([-3.0, -1.0, 1.0, 3.0],device=device)
    mean_z = torch.sum(H*(P@s), axis=2, keepdim = True) - H*(P@s)
    var_z = torch.sum((H**2)*(P@(s*s)-(P@s)**2),axis=2,keepdim=True)-(H**2)*(P@(s*s)-(P@s)**2)+noise_var
    beta = ((torch.unsqueeze(2*H*(y - mean_z), dim=3))@(s-s[0]).view(1,s.shape[0])-(torch.unsqueeze(H**2, dim=3))@(s**2-(s[0])**2).view(1,s.shape[0]))/(2.0*torch.unsqueeze(var_z, dim=3))
    gamma = torch.sum(beta, dim=1,keepdim=True)
    alpha = gamma - beta
    Post_P = F.softmax(alpha, dim=3)
    P1 = (1-delta)*Post_P + delta*P
    x_p = s[torch.argmax(gamma, dim=3)]
    return x_p, P1

def MS_16QAM_layer( y, H, noise_var, P, delta, device):
    s = torch.tensor([-3.0, -1.0, 1.0, 3.0],device=device)
    mean_z = torch.sum(H*(P@s), axis=2, keepdim = True) - H*(P@s)
    var_z = torch.sum((H**2)*(P@(s*s)-(P@s)**2),axis=2,keepdim=True)-(H**2)*(P@(s*s)-(P@s)**2)+noise_var
    beta = ((torch.unsqueeze(2*H*(y - mean_z), dim=3))@(s-s[0]).view(1,s.shape[0])-(torch.unsqueeze(H**2, dim=3))@(s**2-(s[0])**2).view(1,s.shape[0]))/(2.0*torch.unsqueeze(var_z, dim=3))
    gamma = torch.sum(beta, dim=1,keepdim=True)
    alpha = gamma - beta
    Post_P = torch.exp(alpha- torch.amax(alpha, dim=3, keepdim=True))
    P1 = (1-delta)*Post_P + delta*P
    x_p = s[torch.argmax(gamma, dim=3)]
    return x_p, P1

def CHEMP_16QAM_layer(z, J, noise_var, P, delta, device):
    s = torch.tensor([[-3.0, -1.0, 1.0, 3.0]], device=device).T
    J_diag = torch.unsqueeze(torch.diagonal(J, offset=0, dim1=1, dim2=2), dim=2)
    mean_v = torch.sum(J*torch.transpose(P@s,1,2),dim=2,keepdim=True) - J_diag*(P@s)
    var_v = torch.sum((J**2)*torch.transpose(P@(s*s)-(P@s)**2, 1, 2),dim=2,keepdim=True)+noise_var-(J_diag**2)*(P@(s*s)-(P@s)**2)
    L = (2*(z-mean_v)-J_diag*(s+s[0]).T)*(J_diag*(s-s[0]).T)/(2.0*var_v)
    Post_P = F.softmax(L, dim=2)
    P1 = (1-delta)*Post_P + delta*P
    x_p = s[torch.argmax(P1, dim = 2)]
    return x_p, P1

def sMPD_16QAM_layer(z, J, P, w, b, delta, var_j, device):
    s = torch.tensor([[-3.0, -1.0, 1.0, 3.0]], device=device).T
    J_diag = torch.unsqueeze(torch.diagonal(J, offset=0, dim1=1, dim2=2), dim=2)
    mean_v = torch.sum(J*torch.transpose(P@s,1,2),dim=2,keepdim=True) - J_diag*(P@s) 
    L = (2*(z-mean_v)-J_diag*(s+s[0]).T)*(J_diag*(s-s[0]).T)/(2.0*var_j)
    Post_P = w*torch.exp(L - torch.max(L,dim=2,keepdim=True).values) + b
    P1 = (1-delta)*Post_P + delta*P
    x_p = s[torch.argmax(P1, dim = 2)]
    return x_p, P1

