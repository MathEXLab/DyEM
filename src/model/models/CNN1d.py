import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.in_length = cfg.in_length
        self.in_channels = cfg.in_channels
        self.hidden_size = cfg.hidden_size
        self.n_layers = cfg.n_layers
        self.out_channels = cfg.out_channels
        self.kernel_size = cfg.kernel_size

        self.conv_input = nn.Conv1d(in_channels=self.in_channels, out_channels=self.hidden_size, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2)    # remain sim dim
        self.relu = nn.ReLU()
        self.model = nn.Sequential()
        # add cnn input
        self.model.add_module('conv_input', self.conv_input)
        self.model.add_module('relu', self.relu)

        for i in range(self.n_layers-1):
            self.model.add_module('conv{}'.format(i), nn.Conv1d(in_channels=self.hidden_size, out_channels=self.hidden_size, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2))
            self.model.add_module('relu{}'.format(i), nn.ReLU())

        self.fc1 = nn.Linear(self.hidden_size*self.in_length, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.out_channels)

        
    def forward(self, x):
        # (batch, n_t, n_var) -> (batch, n_var, nt)
        x = x.permute(0, 2, 1)
        x = self.model(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

if __name__ == '__main__':
    encoder = Model(in_length = 5, in_channels=3, hidden_size=16,out_channels=3, kernel_size=3,)
    x = torch.randn(16, 3,5)    # [batch, n_var, nt]
    print(encoder(x).shape)
    # output: 'torch.Size([10, 10])