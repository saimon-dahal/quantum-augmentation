import torch.nn as nn


class DAEM_MLP(nn.Module):
    """Residual MLP that learns the noisy → ideal correction."""

    def __init__(self, n_obs: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_obs, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_obs),
        )

    def forward(self, x):
        return x + self.net(x)
