import torch
import torch.nn as nn
import torch.optim as optim

X = torch.randn(300, 2)
y = (2 * X[:, 0] - 3 * X[:, 1] + 1).unsqueeze(1)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        return self.fc2(h)

model = MLP()
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

for epoch in range(300):
    optimizer.zero_grad()
    pred = model(X)
    loss = criterion(pred, y)
    loss.backward()
    optimizer.step()
    if epoch % 50 == 0:
        print(epoch, loss.item())