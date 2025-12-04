import torch
import torch.nn as nn
import torch.optim as optim
from dataset import load_data
from model import RNNPredictor
from visualize import plot_loss, plot_prediction, plot_hidden

def train_model():
    X_train, y_train, X_test, y_test = load_data()
    model = RNNPredictor()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    losses=[]
    for epoch in range(100):
        optimizer.zero_grad()
        y_pred, h_all = model(X_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.6f}")

    model.eval()
    with torch.no_grad():
        y_pred_test, h_all_test = model(X_test)

    torch.save(model.state_dict(), "rnn.pth")

    plot_loss(losses)
    plot_prediction(y_test, y_pred_test)
    plot_hidden(h_all_test)

    return model

if __name__ == "__main__":
    train_model()