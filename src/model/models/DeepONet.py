import torch
import torch.nn as nn
import numpy as np
from ..layers.deeponet_components import *



branch_dict = {
    'mlp': MLP
}

trunk_dict = {  
    'mlp': MLP
}

class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        if cfg.branch.model_name != 'mlp' or cfg.trunk.model_name != 'mlp':
            raise NotImplementedError('Branch and trunk can only be mlp for now')

        # mlp
        self.branch = branch_dict[cfg.branch.model_name](cfg.branch)
        self.trunk = trunk_dict[cfg.trunk.model_name](cfg.trunk)
        
    def forward(self, x):
        # x/x_func: torch.Tensor, [batch_size, n_space*nvar, x_t]
        # x_coor: torch.Tensor, [n_var*n_x, n_coor_dim], (3,1) for lorenz
        # require x to have 3 dimensions
        x_branch, x_trunk = x
        
        x_trunk = x_trunk[0,...]
        x_trunk = x_trunk.flatten()
        nx = x_trunk.shape[0]

        assert x_branch.dim() == 3, 'x should have 3 dimensions (batch_size, n_space*nvar, n_t)'

        x_branch = x_branch.reshape(x_branch.shape[0], -1)  # branch net input, (batch_size, ...)'

        # set dtype to float32
        x_branch = x_branch.float()
        x_trunk = x_trunk.float()

        x_branch = self.branch(x_branch)  #(batch_size, out_dim)
        x_trunk = self.trunk(x_trunk)   #(out_dim*n_var)
        x_trunk = x_trunk.reshape(nx,-1)

        assert x_trunk.shape[-1] == x_trunk.shape[-1], 'x_func and x_loc should have the same last dimension'
        x = torch.einsum('bi,ni->bn', x_branch, x_trunk)
        x =x.unsqueeze(1)

        return x
