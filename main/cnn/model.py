import torch
import torch.nn as nn

class TinyCNN(nn.Module):
    """
    Tiny CNN:
        - 1 Conv Layer
        - out_channels=4
        - kernel_size=3
    """
    def __init__(self, in_channels=1, out_channels=4, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(out_channels * 26 * 26, 2)

    def forward(self, x):
        x = self.relu(self.conv(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)