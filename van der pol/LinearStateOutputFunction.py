from torch import nn
import numpy as np
import torch

class linear_state_net(nn.Module):
    def __init__(self, nx, nu, bias=True):
        super(linear_state_net, self).__init__()
        self.nu = tuple() if nu is None else ((nu,) if isinstance(nu,int) else nu) #None=1 input
        self.net = nn.Linear(in_features=nx+np.prod(self.nu,dtype=int), out_features=nx,bias=bias)

    def forward(self, x, u):
        net_in = torch.cat([x,u.view(u.shape[0],-1)],axis=1)
        return self.net(net_in)
    
class linear_output_net(nn.Module):
    def __init__(self, nx, ny, bias=True):
        super(linear_output_net, self).__init__()
        self.ny = tuple() if ny is None else ((ny,) if isinstance(ny,int) else ny)
        self.net = nn.Linear(in_features=nx, out_features=np.prod(self.ny,dtype=int),bias=bias) 
        
    def forward(self, x):
        return self.net(x).view(*((x.shape[0],)+self.ny))