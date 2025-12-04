import torch
import torch.nn as nn

class RNNPredictor(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, h_t = self.rnn(x)  # (B, T, H)
        last_h = out[:, -1, :]
        y = self.fc(last_h)
        return y, out