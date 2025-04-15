import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from omegaconf import OmegaConf
import math
     

activation_dict = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'sigmoid': nn.Sigmoid(),
    'leaky_relu': nn.LeakyReLU(),
    'gelu': nn.GELU(),
    'selu': nn.SELU(),
    'elu': nn.ELU(),
    'none': nn.Identity()
}

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    
class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.input_dim: int = cfg.input_dim  
        self.in_len: int = cfg.in_len
        self.out_len: int = cfg.out_len
        self.d_model: int = cfg.d_model
        self.nhead: int = cfg.n_head
        self.num_encoder_layers: int = cfg.n_encoder_layers
        # self.num_decoder_layers: int = cfg.n_decoder_layers
        self.dim_feedforward: int = cfg.dim_feedforward
        self.dropout: float = cfg.dropout
        self.output_dim: int = cfg.output_dim

        if hasattr(cfg, 'activation'):
            self.activation = activation_dict[cfg.activation]
        else:
            self.activation: str = 'relu'

        if hasattr(cfg, 'use_pos'):
            self.use_pos = cfg.use_pos  
        else:
            self.use_pos = False

        self.encoder = nn.Linear(self.input_dim, self.d_model)
        # add positional encoding
        if self.use_pos:
            self.pos_encoder = PositionalEncoding(d_model=self.d_model, dropout=self.dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead,
                                                   dim_feedforward=self.dim_feedforward,
                                                   dropout=self.dropout, batch_first=True,
                                                   activation=self.activation)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_encoder_layers, norm=nn.LayerNorm(self.d_model),)
        self.embedding = nn.Linear(self.in_len, self.out_len)
        self.decoder = nn.Linear(self.d_model, self.output_dim)

        # try other decoder?
    # def _generate_square_subsequent_mask(self, sz):
    #     mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    #     mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    #     return mask

        # init weights?
    #     self.init_weights()

    # def init_weights(self):
    #     initrange = 0.1    
    #     self.decoder.bias.data.zero_()
    #     self.decoder.weight.data.uniform_(-initrange, initrange)



    def forward(self, x):
        # to device
        x = self.encoder(x)
        if self.use_pos:
            x = self.pos_encoder(x)
        x = self.transformer_encoder(x)

        # swap axis 1 and 2 - for out seq length
        x = x.permute(0, 2, 1)
        x = self.embedding(x)
        x = x.permute(0, 2, 1)

        x = self.decoder(x)

        return x
    
