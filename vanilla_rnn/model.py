import torch
from torch.nn import init
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, vocab_size, device):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.device = device

        self.embedding = nn.Embedding(vocab_size, input_size)
        self.W_xh = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_hh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_h = nn.Parameter(torch.Tensor(hidden_size))
        self.cls = nn.Parameter(torch.Tensor(hidden_size, output_size))
        self.cls_bias = nn.Parameter(torch.Tensor(output_size))

        init.xavier_uniform_(self.W_xh)
        init.xavier_uniform_(self.W_hh)
        init.xavier_uniform_(self.cls)
        init.zeros_(self.cls_bias)
        init.zeros_(self.b_h)

    def forward(self, x):
        h_t = torch.zeros(
            x.shape[0],
            self.hidden_size,
            requires_grad=False
        ).to(self.device)
        
        num_steps = x.shape[1]
        x = self.embedding(x)
        
        for t in range(num_steps):
            x_t = x[:, t, :]
            h_t = torch.tanh(x_t @ self.W_xh + h_t @ self.W_hh + self.b_h)

        output = h_t @ self.cls + self.cls_bias

        return output