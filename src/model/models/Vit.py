import torch
from vit_pytorch import ViT
import lightning as L
import numpy as np



class Model(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.width = cfg.width
        self.height = cfg.height
        self.patch_width = cfg.patch_width
        self.patch_height = cfg.patch_height
        self.in_channels = cfg.input_dim    # input steps
        self.output_dim = cfg.output_dim    # output steps
        self.output_size = cfg.output_dim * cfg.width * cfg.height
        self.laten_dim = cfg.latent_dim
        self.depth = cfg.n_layers
        self.heads = cfg.heads
        
        if hasattr(cfg, 'mlp_dim'):
            self.mlp_dim = cfg.mlp_dim
        else:
            self.mlp_dim = self.laten_dim
        # dropout, optional
        if hasattr(cfg, 'dropout'):
            self.dropout = cfg.dropout
        else:
            self.dropout = 0.1
        if hasattr(cfg, 'emb_dropout'):
            self.emb_dropout = cfg.emb_dropout
        else:
            self.emb_dropout = 0.1

        model = ViT(
                    image_size = (self.width, self.height),
                    patch_size = (self.patch_width, self.patch_height),
                    channels = self.in_channels,
                    num_classes = self.output_size,
                    dim = self.laten_dim,
                    depth = self.depth,
                    heads = self.heads,
                    mlp_dim = self.mlp_dim,
                    dropout = self.dropout,
                    emb_dropout = self.emb_dropout
                    )
        self.model = model
        # self.save_hyperparameters()
        
    def forward(self, x):
        '''
        x: (batch_size, channel, width, height)
        '''
        x = self.model(x)
        x = x.view(-1, self.output_dim, self.width, self.height)
        return x