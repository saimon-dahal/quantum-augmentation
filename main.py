import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, pauli_error

NUM_QUBITS = 2
SHOTS = 4096
BASIS = ["00", "01", "10", "11"]

np.random.seed(42)
torch.manual_seed(42)


def create_asymmetric_noise_model():
    """
    This function defines how noise behaves on the hardware
    """
    nm = NoiseModel()

    # for qubit 0: worse X-type noise
    nm.add_quantum_error(
        pauli_error([("X", 0.06), ("I", 0.94)]),
        ["rx", "ry", "rz"],
        [0],
    )

    # for qubit 1: milder Y-type noise
    nm.add_quantum_error(
        pauli_error([("Y", 0.02), ("I", 0.98)]),
        ["rx", "ry", "rz"],
        [1],
    )

    # Control -> target (0 -> 1) is noisier
    nm.add_quantum_error(
        depolarizing_error(0.07, 2),
        "cx",
        [0, 1],
    )

    # Reverse direction (1 -> 0): different noise strength
    nm.add_quantum_error(
        depolarizing_error(0.03, 2),
        "cx",
        [1, 0],
    )

    return nm


# noisy simulator and ideal simulator
noisy_backend = AerSimulator(noise_model=create_asymmetric_noise_model())
ideal_backend = AerSimulator()


def build_target(theta, phi):
    """
    This represents the real quantum process that we are trying to correct
    """
    qc = QuantumCircuit(2, 2)

    # these gates are the skeleton
    qc.rx(theta, 0)
    qc.ry(phi, 1)

    # entangling gate
    qc.cx(0, 1)

    qc.rz(theta / 2, 0)
    qc.ry(-phi / 2, 1)

    qc.measure([0, 1], [0, 1])
    return qc


def build_fiducial(theta, phi):
    """
    Fiducial circuit with same skelton as target.
    All single qubit gates becomes identity (but noisy)
    """
    qc = QuantumCircuit(NUM_QUBITS, NUM_QUBITS)

    qc.rx(theta / 2, 0)
    qc.rx(-theta / 2, 0)

    qc.ry(phi / 2, 1)
    qc.ry(-phi / 2, 1)

    # keep CNOT unchanged to preserve dominant noise
    qc.cx(0, 1)

    qc.rz(theta / 4, 0)
    qc.rz(-theta / 4, 0)

    qc.ry(-phi / 4, 1)
    qc.ry(phi / 4, 1)

    qc.measure([0, 1], [0, 1])
    return qc


def counts_to_probs(counts):
    """Convert raw counts into probability distribution"""
    return np.array([counts.get(b, 0) / SHOTS for b in BASIS])


def simulate(qc, backend):
    result = backend.run(qc, shots=SHOTS).result()
    return counts_to_probs(result.get_counts())


def generate_dataset(n_samples=300):
    """
    Generate dataset of (noisy, ideal) pairs using fiducial circuits
    """

    X, Y = [], []

    for _ in range(n_samples):
        theta = np.random.uniform(0, np.pi)
        phi = np.random.uniform(0, np.pi)

        fid = build_fiducial(theta, phi)

        noisy = simulate(fid, noisy_backend)
        ideal = simulate(fid, ideal_backend)

        X.append(noisy)
        Y.append(ideal)

    X = np.array(X)
    Y = np.array(Y)

    if X.shape != Y.shape:
        raise RuntimeError("Dataset mismatch between noisy and ideal.")

    return X, Y


class MLP(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=32, output_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        return self.net(x)


def train_model(model, X, Y, epochs=200, lr=1e-3):
    # convert to tensors
    X_t = torch.tensor(X, dtype=torch.float32)
    Y_t = torch.tensor(Y, dtype=torch.float32)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.KLDivLoss(reduction="batchmean")
    # loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        optimizer.zero_grad()

        pred = model(X_t)
        loss = loss_fn(torch.log(pred + 1e-12), Y_t)
        # loss = loss_fn(pred, Y_t)

        if torch.isnan(loss):
            raise RuntimeError("NaN in loss")

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1}, loss={loss.item():.6f}")
    return model


# apply mitigation on target circuit
def mitigate(theta, phi, model):

    target = build_target(theta, phi)

    noisy = simulate(target, noisy_backend)
    ideal = simulate(target, ideal_backend)

    x = torch.tensor(noisy, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        mitigated = model(x).squeeze(0).numpy()

    return noisy, ideal, mitigated


def evaluate(model, n_tests=100):
    rows = []

    mae_noisy_list = []
    mae_mitigated_list = []

    improved_count = 0

    for _ in range(n_tests):
        theta = np.random.uniform(0, np.pi)
        phi = np.random.uniform(0, np.pi)

        noisy, ideal, mitigated = mitigate(theta, phi, model)

        # mean absolute errors
        mae_noisy = np.mean(np.abs(noisy - ideal))
        mae_mitigated = np.mean(np.abs(mitigated - ideal))

        mae_noisy_list.append(mae_noisy)
        mae_mitigated_list.append(mae_mitigated)

        # check if mitigation improved this sample
        if mae_mitigated < mae_noisy:
            improved_count += 1

        rows.append(
            {
                "noisy": noisy.tolist(),
                "ideal": ideal.tolist(),
                "mitigated": mitigated.tolist(),
                "mae_noisy": mae_noisy,
                "mae_mitigated": mae_mitigated,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv("results.csv", index=False)

    # aggregate metrics
    avg_mae_noisy = np.mean(mae_noisy_list)
    avg_mae_mitigated = np.mean(mae_mitigated_list)

    fraction_improved = improved_count / n_tests

    print(f"MAE Noisy: {avg_mae_noisy:.6f}")
    print(f"MAE Mitigated: {avg_mae_mitigated:.6f}")
    print(f"Fraction Improved: {fraction_improved:.2f}")

    return df


if __name__ == "__main__":
    # generate training data
    X, Y = generate_dataset(500)

    diff = np.abs(X - Y).mean()
    print("Average deviation per sample:", diff)

    # initialise model and optimiser
    model = MLP()
    model = train_model(model, X, Y)
    df = evaluate(model)
