import torch
import torch.nn as nn
import torch.optim as optim

class mlp(nn.Module):
    """
    1-hidden layer MLP for regression.
    Input dim: 3
    Hidden dim: 16
    Output dim: 1
    """

    def __init__(self, in_dim: int = 3, hidden_dim: int = 16, out_dim: int = 1):
        super().__init__()
        self.fc1 = nn.Linear(3, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        return self.fc2(h)