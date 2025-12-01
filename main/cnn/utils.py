import numpy as np
import torch

def generate_circle(size=28, radius=8):
    img = np.zeros((size, size), dtype=np.float32)
    cx, cy = size//2, size//2
    for i in range(size):
        for j in range(size):
            if (i-cx)**2 + (j-cy)**2 <= radius * radius:
                img[i, j] = 1.0
    return img

def generate_cross(size=28, thickness=3):
    img = np.zeros((size, size), dtype=np.float32)
    mid = size // 2
    img[mid-thickness:mid+thickness, :] = 1.0
    img[:, mid-thickness:mid+thickness] = 1.0
    return img

def generate_dataset(n_samples=200):
    X = []
    y = []
    for _ in range(n_samples):
        if np.random.rand() < 0.5:
            X.append(generate_circle())
            y.append(0)
        else:
            X.append(generate_cross())
            y.append(1)
    X = np.array(X)
    X = X[:, None, :, :]
    y = np.array(y)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)