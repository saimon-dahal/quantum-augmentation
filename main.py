import numpy as np
import pandas as pd
from qiskit_ibm_runtime.fake_provider import FakeNairobiV2

from training_data import build_training_dataset, get_pauli_observables

if __name__ == "__main__":
    N_QUBITS = 3
    N_LAYERS = 2
    N_SAMPLES = 50
    SHOTS = 4096  # shots per observable per sample

    noisy_backend = FakeNairobiV2()

    # --- Observables ---
    obs_list, obs_labels = get_pauli_observables(N_QUBITS)
    print(f"Number of observables: {len(obs_labels)}")
    # For n_qubits=3: 9 single-qubit + 27 two-qubit = 36 observables

    # --- Build dataset ---
    print(f"\nBuilding dataset: {N_SAMPLES} samples × {len(obs_labels)} observables")
    print("Training on FIDUCIAL circuit only (never the target).\n")

    X, y = build_training_dataset(
        n_qubits=N_QUBITS,
        n_layers=N_LAYERS,
        n_samples=N_SAMPLES,
        obs_list=obs_list,
        obs_labels=obs_labels,
        noisy_backend=noisy_backend,
        shots=SHOTS,
        seed=42,
    )

    print("\nDataset shapes:")
    print(f"  X (noisy features) : {X.shape}")
    print(f"  y (ideal labels)   : {y.shape}")

    # --- Sanity checks ---
    print("\nSanity checks:")
    print(f"  X range: [{X.min():.3f}, {X.max():.3f}]  (should be in [-1, 1])")
    print(f"  y range: [{y.min():.3f}, {y.max():.3f}]  (should be in [-1, 1])")
    print(f"  Mean |noisy - ideal|: {np.abs(X - y).mean():.4f}")

    # --- Save ---
    np.save("X_train.npy", X)
    np.save("y_train.npy", y)

    df_X = pd.DataFrame(X, columns=[f"noisy_{l}" for l in obs_labels])
    df_y = pd.DataFrame(y, columns=[f"ideal_{l}" for l in obs_labels])
    df = pd.concat([df_X, df_y], axis=1)
    df.to_csv("training_data.csv", index=False)

    print("\nSaved: X_train.npy, y_train.npy, training_data.csv")
    print("\nFirst sample preview:")
    for lbl, nv, iv in zip(obs_labels[:6], X[0, :6], y[0, :6]):
        print(f"  {lbl:<10} noisy={nv:+.4f}  ideal={iv:+.4f}  diff={nv - iv:+.4f}")
