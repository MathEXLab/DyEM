import torch
import torch.nn as nn



class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.in_channels = cfg.in_channels
        self.hidden_size = cfg.hidden_size
        self.out_channels = cfg.out_channels
        self.kernel_size = cfg.kernel_size
        # self.width = cfg.width
        # self.height = cfg.height

        self. model = nn.Sequential(
            nn.Conv2d(self.in_channels, int(self.hidden_size/2), self.kernel_size, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(int(self.hidden_size/2)),
            nn.Conv2d(int(self.hidden_size/2), self.hidden_size, self.kernel_size, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(self.hidden_size),
            nn.Conv2d(self.hidden_size, self.hidden_size, self.kernel_size, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(self.hidden_size),
            nn.Conv2d(self.hidden_size, int(self.hidden_size/2), self.kernel_size, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(int(self.hidden_size/2)),
            nn.Conv2d(int(self.hidden_size/2), self.out_channels, self.kernel_size, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(self.out_channels),
        )
        
    def forward(self, x):
        # (batch, channal, width, height)
        # x = x.permute(0, 2, 1)
        x = self.model(x)

        # residual connection
        # x = x + x_out
        return x
    
