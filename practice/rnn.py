import torch
import torch.nn as nn

rnn = nn.RNN(
    input_size = 5,
    hidden_size = 16,
    batch_first = True
)

x = torch.randn(32, 10, 5)
out, h_T = rnn(x)
print(out.shape, h_T.shape)