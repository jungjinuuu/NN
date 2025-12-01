import torch
import torch.nn as nn
import torch.optim as optim
from model import TinyCNN
from utils import generate_dataset

def train():
    X, y = generate_dataset(n_samples=200)
    model = TinyCNN()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(10):
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        if epoch % 1 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "tinycnn.pth")
    print("Model saved to tinycnn.pth")

    return model, X

if __name__ == "__main__":
    train()