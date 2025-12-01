# ---------- Not Using PyTorch ----------

import numpy as np

def conv2d(x, kernel):
    H, W = x.shape
    kH, kW = kernel.shape
    out = np.zeros((H - kH + 1, W - kW + 1))

    for i in range(H - kH + 1):
        for j in range(W - kW + 1):
            patch = x[i:i+kH, j:j+kW]
            out[i, j] = np.sum(patch * kernel)

    return out

x = np.array([[1,2,3],
            [4,5,6],
            [7,8,9]], dtype=float)

kernel = np.array([[1,0],
                [0,-1]], dtype=float)

print(conv2d(x, kernel))



# ---------- Using PyTorch ----------

import torch
import torch.nn as nn

class simpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(4*26*26, 10)

    def forward(self, x):
        x = self.relu(self.conv(x))
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out