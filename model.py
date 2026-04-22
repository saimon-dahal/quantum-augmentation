import torch.nn as nn


class DAEM_MLP(nn.Module):
    """
    MLP that learns the noisy -> ideal mapping for Pauli expectation values.

    Input/output dim = number of observables (e.g. 21 for 3 qubits).
    No Softmax — expectation values live in [-1, +1], not a probability simplex.
    Residual connection: the model learns the CORRECTION, not the full mapping.
    """

    def __init__(self, n_obs, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_obs, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_obs),
            # FIX: NO Softmax — outputs are expectation values in [-1, 1]
        )

    def forward(self, x):
        # FIX: residual — learn the noise correction, not the full output
        return x + self.net(x)
