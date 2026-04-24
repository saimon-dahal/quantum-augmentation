import json

import numpy as np
import torch
import torch.nn as nn
from qiskit_ibm_runtime.fake_provider import FakeNairobiV2
from torch.utils.data import DataLoader, TensorDataset

from circuit import VQECircuit
from model import DAEM_MLP
from training_data import (
    build_training_dataset,
    compute_noisy_expvals,
    get_pauli_observables,
    ideal_expectation_statevector,
    make_noisy_sim,
)

N_QUBITS = 3
N_LAYERS = 2
N_SAMPLES = 200
SHOTS = 4096
EPOCHS = 300
LR = 1e-3
HIDDEN = 128
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)


def main():
    # Step 1 — Noisy simulator
    print(f"Building noise model from FakeNairobiV2...")
    fake_backend = FakeNairobiV2()
    noisy_sim = make_noisy_sim(fake_backend)
    obs_list, obs_labels = get_pauli_observables(N_QUBITS)
    n_obs = len(obs_list)
    print(f"Observables: {n_obs} ({N_QUBITS}-qubit Paulis)\n")

    # Step 2 — Training data from fiducial circuits
    print(f"Generating {N_SAMPLES} training samples from fiducial circuits...")
    X, y = build_training_dataset(
        n_qubits=N_QUBITS,
        n_layers=N_LAYERS,
        n_samples=N_SAMPLES,
        obs_list=obs_list,
        noisy_sim=noisy_sim,
        shots=SHOTS,
        seed=SEED,
    )
    print(f"Training data: X={X.shape}, y={y.shape}\n")

    # Step 3 — Train MLP
    print(f"Training MLP (epochs={EPOCHS}, lr={LR}, hidden={HIDDEN})...")
    X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    y_t = torch.tensor(y, dtype=torch.float32).to(DEVICE)

    n_train = int(0.8 * len(X_t))
    X_tr, X_val = X_t[:n_train], X_t[n_train:]
    y_tr, y_val = y_t[:n_train], y_t[n_train:]

    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=32, shuffle=True)
    model = DAEM_MLP(n_obs=n_obs, hidden_dim=HIDDEN).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val), y_val).item()
        if epoch % 50 == 0 or epoch == 1:
            train_loss = criterion(model(X_tr), y_tr).item() if epoch % 50 == 0 else val_loss
            print(f"  Epoch {epoch:4d}/{EPOCHS}  val_loss={val_loss:.5f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    torch.save(best_state, "daem_model.pt")
    print(f"  Best val_loss: {best_val_loss:.5f} — saved to daem_model.pt\n")

    # Step 4 — Run target VQE circuit
    print("Running inference on target VQE circuit...")
    theta_target = rng.uniform(0, 2 * np.pi, size=4 * N_QUBITS * N_LAYERS)
    vqe = VQECircuit(N_QUBITS, N_LAYERS, theta_target)
    target_circuit = vqe.build_target()

    noisy_expvals = compute_noisy_expvals(target_circuit, obs_list, noisy_sim, shots=SHOTS)
    ideal_expvals = np.array(
        [ideal_expectation_statevector(target_circuit, obs) for obs in obs_list]
    )

    model.eval()
    with torch.no_grad():
        x_in = torch.tensor(noisy_expvals, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        mitigated_expvals = model(x_in).squeeze(0).cpu().numpy()

    # Step 5 — Evaluate
    mae_noisy = np.abs(noisy_expvals - ideal_expvals).mean()
    mae_mitigated = np.abs(mitigated_expvals - ideal_expvals).mean()
    improvement = (mae_noisy - mae_mitigated) / mae_noisy * 100

    results = {
        "config": {
            "n_qubits": N_QUBITS,
            "n_layers": N_LAYERS,
            "n_samples": N_SAMPLES,
            "shots": SHOTS,
            "epochs": EPOCHS,
            "lr": LR,
            "hidden_dim": HIDDEN,
            "seed": SEED,
            "backend": fake_backend.name,
        },
        "training": {
            "best_val_loss": float(best_val_loss),
        },
        "evaluation": {
            "mae_noisy": float(mae_noisy),
            "mae_mitigated": float(mae_mitigated),
            "improvement_pct": float(improvement),
        },
        "per_observable": [],
    }

    header = f"{'Observable':<12} {'Noisy':>8} {'Mitigated':>10} {'Ideal':>8} {'Err_noisy':>10} {'Err_mitig':>10}"
    print(header)
    print("-" * len(header))
    for lbl, nv, mv, iv in zip(obs_labels, noisy_expvals, mitigated_expvals, ideal_expvals):
        print(f"{lbl:<12} {nv:+8.4f} {mv:+10.4f} {iv:+8.4f} {abs(nv - iv):10.4f} {abs(mv - iv):10.4f}")
        results["per_observable"].append({
            "label": lbl,
            "noisy": float(nv),
            "mitigated": float(mv),
            "ideal": float(iv),
        })

    print(f"\nMAE  noisy:     {mae_noisy:.4f}")
    print(f"MAE  mitigated: {mae_mitigated:.4f}")
    print(f"Improvement:    {improvement:.1f}%")

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
