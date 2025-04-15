import torch
import torch.nn as nn
import numpy as np
from ..layers.deeponet_components import *



branch_dict = {
    'mlp': MLP,
    'cnn': CNN
}

trunk_dict = {  
    'mlp': MLP
}

class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.branch = branch_dict[cfg.branch.model_name](cfg.branch)
        self.trunk = trunk_dict[cfg.trunk.model_name](cfg.trunk)
        
    def forward(self, x):
        # x/x_func: torch.Tensor, [batch_size, n_space*nvar, x_t]
        # x_coor: torch.Tensor, [n_var*n_x, n_coor_dim], (3,1) for lorenz
        # require x to have 3 dimensions
        x_branch, x_trunk = x # (b,c,width,height), (b,width,height,n_coors)
        
        x_trunk = x_trunk[0,...]    # remove batch
        n_x,n_y,_ = x_trunk.shape
        # x_trunk = x_trunk.reshape(-1)  # flatten all the dimensions

        # set dtype to float32
        x_branch = x_branch.float()
        x_trunk = x_trunk.float()

        # x_branch = x_branch.reshape(x_branch.shape[0], -1)  # branch net input, (batch_size, ...)'

        x_branch = self.branch(x_branch)  #(batch_size, out_dim)
        x_trunk = self.trunk(x_trunk)   #(out_dim*n_var)
        # x_trunk = x_trunk.reshape(-1,1)

        assert x_trunk.shape[-1] == x_trunk.shape[-1], 'x_func and x_loc should have the same last dimension'
        x = torch.einsum('bi, xyi->bxy', x_branch, x_trunk)
        x =x.unsqueeze(1)

        # recover shape
        # x = x.reshape(n_x,n_y,-1) 


        return x
