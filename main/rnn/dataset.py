import numpy as np
import torch

def generate_sine(n_points=500, period=50):
    t = np.arange(n_points)
    x = np.sin(2 * np.pi * t / period)
    return x

def make_dataset(series, window_size=20):
    X = []
    y = []
    for i in range(len(series) - window_size):
        X.append(series[i:i+window_size])
        y.append(series[i+window_size])
    X = np.array(X)
    y = np.array(y)

    X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
    return X, y

def load_data(window_size=20):
    series = generate_sine()
    X, y = make_dataset(series, window_size=window_size)

    train_size = int(0.8 * len(X))
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]

    return X_train, y_train, X_test, y_test