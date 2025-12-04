import matplotlib.pyplot as plt
import torch

def plot_loss(losses):
    plt.figure(figsize=(6,4))
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.tight_layout()
    plt.savefig("loss_curve.png")
    plt.close()

def plot_prediction(y_true, y_pred):
    y_true = y_true.squeeze().cpu().numpy()
    y_pred = y_pred.squeeze().cpu().numpy()

    plt.figure(figsize=(8,4))
    plt.plot(y_true, label="True")
    plt.plot(y_pred, label="Predicted")
    plt.legend()
    plt.title("Next-Step Prediction")
    plt.tight_layout()
    plt.savefig("prediction.png")
    plt.close()

def plot_hidden(h_all):
    # h_all: (B, T, H)
    h = h_all[0].detach().cpu().numpy()
    dim0 = h[:, 0]

    plt.figure(figsize=(8,4))
    plt.plot(dim0)
    plt.title("Hidden State (dimension 0)")
    plt.xlabel("Time Step")
    plt.tight_layout()
    plt.savefig("hidden_state.png")
    plt.close()
