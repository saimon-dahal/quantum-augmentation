import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 32), nn.ReLU(), nn.Linear(32, dim), nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)


class DAEM_MLP_Residual(nn.Module):
    def __init__(self, n_qubits, hidden_dim=64):
        super().__init__()
        dim = 2**n_qubits
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        correction = self.net(x)
        return self.softmax(x + correction)  # learn residual, not full mapping
