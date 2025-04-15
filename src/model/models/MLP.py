import torch
import torch.nn as nn



class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.input_dim = cfg.input_dim
        self.hidden_dim = cfg.hidden_dim
        self.n_layer = cfg.n_layer
        self.output_dim = cfg.output_dim

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, int(self.hidden_dim/2)),
            nn.ReLU(),
            nn.Linear(int(self.hidden_dim/2), self.hidden_dim),
            nn.ReLU()
        )   
        for i in range(self.n_layer-1):
            self.model.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            self.model.append(nn.ReLU())
        self.model.append(nn.Linear(self.hidden_dim, self.output_dim))

    def forward(self, x):
        x = self.model(x)
        return x