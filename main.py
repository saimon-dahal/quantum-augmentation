"""
DAEM Minimal Example
=====================
1. Generate (noisy, ideal) training pairs from FIDUCIAL circuits
2. Train MLP to learn the noisy -> ideal correction
3. Run the TARGET VQE circuit through noisy sim -> get noisy expvals
4. Use trained MLP to denoise those expvals
5. Compare: noisy vs mitigated vs ideal
"""

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

# ── Config ────────────────────────────────────────────────────────────────────
N_QUBITS = 3
N_LAYERS = 2
N_SAMPLES = 200  # training samples (each uses a fresh random theta)
SHOTS = 4096
EPOCHS = 300
LR = 1e-3
HIDDEN = 128
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)


# ── Step 1: Build noisy simulator ─────────────────────────────────────────────
print("=" * 60)
print("  DAEM Minimal Example")
print("=" * 60)

fake_backend = FakeNairobiV2()
noisy_sim = make_noisy_sim(fake_backend)
print(f"\n[1] Noise model built from {fake_backend.name}")

obs_list, obs_labels = get_pauli_observables(N_QUBITS)
n_obs = len(obs_list)
print(f"[2] Observables: {n_obs}  ({N_QUBITS}-qubit single + two-qubit Paulis)")


# ── Step 2: Generate training data from fiducial circuits ─────────────────────
print(f"\n[3] Generating {N_SAMPLES} training samples from fiducial circuits...")
print("    (each sample = fresh random theta -> fiducial noisy + ideal expvals)\n")

X, y = build_training_dataset(
    n_qubits=N_QUBITS,
    n_layers=N_LAYERS,
    n_samples=N_SAMPLES,
    obs_list=obs_list,
    noisy_sim=noisy_sim,
    shots=SHOTS,
    seed=SEED,
)

print(f"\n  X (noisy) shape : {X.shape}")
print(f"  y (ideal) shape : {y.shape}")
print(f"  X range         : [{X.min():.3f}, {X.max():.3f}]")
print(f"  y range         : [{y.min():.3f}, {y.max():.3f}]")
print(f"  Mean |X - y|    : {np.abs(X - y).mean():.4f}  (training noise level)")

np.save("X_train.npy", X)
np.save("y_train.npy", y)
print("  Saved X_train.npy, y_train.npy")


# ── Step 3: Train the MLP ──────────────────────────────────────────────────────
print(f"\n[4] Training DAEM MLP  (epochs={EPOCHS}, lr={LR}, hidden={HIDDEN})")

X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
y_t = torch.tensor(y, dtype=torch.float32).to(DEVICE)

# 80/20 train/val split
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

    if epoch % 50 == 0 or epoch == 1:
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val), y_val).item()
            train_loss = criterion(model(X_tr), y_tr).item()
        print(
            f"  Epoch {epoch:4d}/{EPOCHS}  train_loss={train_loss:.5f}  val_loss={val_loss:.5f}"
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

model.load_state_dict(best_state)
torch.save(best_state, "daem_model.pt")
print(f"\n  Best val_loss: {best_val_loss:.5f}  (model saved to daem_model.pt)")


# ── Step 4: Run VQE target circuit and measure noisy expvals ──────────────────
print("\n[5] Inference on TARGET VQE circuit")

# Use a fixed theta for the target (representing "the circuit we want to run")
theta_target = rng.uniform(0, 2 * np.pi, size=4 * N_QUBITS * N_LAYERS)
vqe = VQECircuit(N_QUBITS, N_LAYERS, theta_target)
target_circuit = vqe.build_target()

print("  Measuring noisy expvals from target circuit...")
noisy_expvals = compute_noisy_expvals(target_circuit, obs_list, noisy_sim, shots=SHOTS)

print("  Computing ideal expvals from target circuit (Statevector)...")
ideal_expvals = np.array(
    [ideal_expectation_statevector(target_circuit, obs) for obs in obs_list]
)

print("  Applying MLP mitigation...")
model.eval()
with torch.no_grad():
    x_in = torch.tensor(noisy_expvals, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    mitigated_expvals = model(x_in).squeeze(0).cpu().numpy()


# ── Step 5: Evaluate ──────────────────────────────────────────────────────────
print("\n[6] Results")
print("=" * 60)

mae_noisy = np.abs(noisy_expvals - ideal_expvals).mean()
mae_mitigated = np.abs(mitigated_expvals - ideal_expvals).mean()
improvement = (mae_noisy - mae_mitigated) / mae_noisy * 100

print(f"\n  MAE (noisy vs ideal)    : {mae_noisy:.4f}")
print(f"  MAE (mitigated vs ideal): {mae_mitigated:.4f}")
print(f"  Improvement             : {improvement:.1f}%")

print(
    f"\n  {'Observable':<12} {'Noisy':>8} {'Mitigated':>10} {'Ideal':>8} {'Err_noisy':>10} {'Err_mitig':>10}"
)
print("  " + "-" * 62)
for lbl, nv, mv, iv in zip(
    obs_labels[:15], noisy_expvals[:15], mitigated_expvals[:15], ideal_expvals[:15]
):
    print(
        f"  {lbl:<12} {nv:+8.4f} {mv:+10.4f} {iv:+8.4f} {abs(nv - iv):10.4f} {abs(mv - iv):10.4f}"
    )
if n_obs > 15:
    print(f"  ... ({n_obs - 15} more observables)")

print("\n" + "=" * 60)
if improvement > 0:
    print(f"  DAEM reduced error by {improvement:.1f}%  ✓")
else:
    print(f"  DAEM did not improve (error increased by {-improvement:.1f}%)")
    print("  -> Try increasing N_SAMPLES or EPOCHS")
print("=" * 60)
