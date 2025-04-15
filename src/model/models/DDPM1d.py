import torch
import torch.nn as nn
from ..layers.denoise_diffusion_components import *


class Model(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.seq_length = cfg.seq_length
        self.timesteps = cfg.timesteps  # difussion steps, default 1000
        self.objective = 'pred_noise'
        self.unet_dim = cfg.unet_dim
        self.dim_mults = tuple(cfg.dim_mults)
        self.channels = cfg.in_channels
        self.out_length = cfg.out_length

        self.model = Unet1D(
                                dim = self.unet_dim,
                                dim_mults = self.dim_mults,
                                channels = self.channels,
                                out_length = self.out_length,
                                in_length = self.seq_length
                            )
        # input (N,C,seq_length)

        self.diffusion = GaussianDiffusion1D(
                                                self.model,
                                                seq_length = self.seq_length,
                                                timesteps = self.timesteps,
                                                objective = self.objective
                                            )
    def forward(self,x):
        # x: (N,seq_length,c)
        x = x.permute(0,2,1)
        return self.diffusion(x)