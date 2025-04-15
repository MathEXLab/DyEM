import torch
import torch.nn as nn

from ..layers.fno_components.models.fno import FNO


class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.modes_x = cfg.modes_x
        self.modes_y = cfg.modes_y
        self.in_channels = cfg.in_channels
        self.hidden_channels = cfg.hidden_channels   
        self.out_channels = cfg.out_channels
        self.model = FNO(n_modes=(self.modes_x, self.modes_y),
                         in_channels=self.in_channels,
                         hidden_channels=self.hidden_channels,
                         out_channels=self.out_channels)

    def forward(self, x):
        return self.model(x)
