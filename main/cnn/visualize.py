import matplotlib.pyplot as plt
import torch
from model import TinyCNN

def visualize_filters():
    model = TinyCNN()
    model.load_state_dict(torch.load("tinycnn.pth"))

    weight = model.conv.weight.data.cpu()

    fig, axes = plt.subplots(1, 4, figsize=(12,3))
    for i in range(4):
        axes[i].imshow(weight[i, 0, :, :], cmap="hot")
        axes[i].set_title(f"Filter {i}")
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig("filters.png", dpi=200)
    plt.show()

if __name__ == "__main__":
    visualize_filters()