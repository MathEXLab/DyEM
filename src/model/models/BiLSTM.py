import torch
import torch.nn as nn



# BiLSTM
class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Hidden dimensions
        self.hidden_dim = cfg.hidden_dim

        # Number of hidden layers
        self.layer_dim = cfg.layer_dim

        # Building your LSTM
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(cfg.input_dim, cfg.hidden_dim, cfg.layer_dim, batch_first=True, bidirectional=True)

        # Readout layer
        self.fc = nn.Linear(cfg.hidden_dim * 2, cfg.output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim * 2, x.size(0), self.hidden_dim).requires_grad_().to(x.device)

        # Initialize cell state
        c0 = torch.zeros(self.layer_dim * 2, x.size(0), self.hidden_dim).requires_grad_().to(x.device)

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Index hidden state of last time step
        out = self.fc(out[:, -1, :])
        # out.size() --> 100, 10
        return out.unsqueeze(1)     # shape = (batch_size, 1, output_dim), 1 step prediction
    

if __name__ == '__main__':
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyperparameters
    input_dim = 3
    hidden_dim = 64
    layer_dim = 4
    output_dim = 3

    # Initialize model
    model = Model(input_dim, hidden_dim, layer_dim, output_dim).to(device)

    # Forward pass
    inp = torch.randn(16, 1, 3).to(device)
    out = model(inp)
    print(out.shape)
