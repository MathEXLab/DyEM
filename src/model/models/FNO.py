import torch
import torch.nn as nn

from ..layers.fno_components.models.fno import FNO


class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_modes = cfg.n_modes
        self.in_channels = cfg.in_channels
        self.hidden_channels = cfg.hidden_channels   
        self.out_channels = cfg.out_channels
        self.model = FNO(n_modes=(self.n_modes),
                         hidden_channels=self.hidden_channels,
                         in_channels=self.in_channels,
                         out_channels=self.out_channels)

    def forward(self, x):
        return self.model(x)
