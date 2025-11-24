import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as pyplot
from sklearn.model_selection import train_test_split

from mlp_train_251124 import mlp


def set_seed(seed: int = 42):
    """Make experiments reproducible."""
    torch.manual_seed(seed)


def generate_data(n_samples: int = 500):
    """
    Generate synthetic regression data:
    y = 3*x1 - 2*x2 + 0.5*x3 + eps
    eps ~ N(0, 0.1^2)
    """
    X = torch.randn(n_samples, 3)
    eps = 0.1 * torch.randn(n_samples)
    y = 3 * X[:, 0] - 2 * X[:, 1] + 0.5 * X[:, 2] + eps
    y = y.unsqueeze(1)
    return X, y


def train(model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    num_epochs: int = 500, 
    lr: float = 1e-2,
):
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        # ---- train step ----
        model.train()
        optimizer.zero_grad()
        pred_train = model(X_train)
        loss_train = criterion(pred_train, y_train)
        loss_train.backward()
        optimizer.step()

        # ---- validation step ----
        model.eval()
        with torch.no_grad():
            pred_val = model(X_val)
            loss_val = criterion(pred_val, y_val)
        
        train_losses.append(loss_train.item())
        val_losses.append(loss_val.item())

        if epoch % 50 == 0:
            print(
                f"Epoch {epoch:3d} | "
                f"train_loss = {loss_train.item():.4f} | "
                f"val_loss = {loss_val.item():.4f}"
            )

        return train_losses, val_losses


def plot_results(
    train_losses,
    val_losses,
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    save_prefix: str | None = None,
):
    """Plot loss curve and true vs predicted scatter."""
    # Loss curve
    plt.figure()
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("epoch")
    plt.ylabel("MSE loss")
    plt.title("Loss curve (train vs val)")
    plt.legend()
    if save_prefix is not None:
        plt.savefig(f"{save_prefix}_loss_curve.png", dpi=200)
    plt.show()

    y_true_np = y_true.detach().cpu().numpy()
    y_pred_np = y_pred.detach().cpu().numpy()

    plt.figure()
    plt.scatter(y_true_np, y_pred_np, s=8)
    plt.xlabel("true y")
    plt.ylabel("predicted y")
    plt.title("True vs Predicted")
    if save_prefix is not None:
        plt.savefig(f"{save_prefix}+scatter.png", dpi=200)
    plt.show()


def main():
    set_seed(42)

    # 1. data
    X, y = generate_data(n_samples=500)

    # train/val split (80/20)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. model
    model = mlp(in_dim=3, hidden_dim=16, out_dim=1)

    # 3. training
    train_losses, val_losses = train(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        num_epochs=500,
        lr=1e-2
    )

    # 4. evaluation on full data
    model.eval()
    with torch.no_grad():
        y_pred = model(X)

    # 5. plots
    plot_results(train_losses, val_losses, y, y_pred, save_prefix=None)


if __name__ == "__main__":
    main()