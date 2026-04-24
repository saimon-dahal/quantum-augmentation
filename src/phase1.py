import json
import time
import warnings

import joblib
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from .ansatz import build_ansatz, build_fiducial
from .backend import create_simulators
from .config import (
    N_G_VALUES,
    N_LAYERS,
    N_PROD_STATES,
    N_QUBITS,
    OUTPUT_DIR,
    SEED,
    SHOTS,
    rng,
)
from .hamiltonian import make_hamiltonian
from .measurement import expval, expval_vec
from .observables import make_nn_observables

warnings.filterwarnings("ignore")


def ideal_fiducial_ev(
    init_bits: list[int],
    fid_qc: QuantumCircuit,
    obs_list: list[SparsePauliOp],
) -> np.ndarray:
    """Compute exact expectation values via Statevector simulation."""
    init_qc = QuantumCircuit(N_QUBITS)
    for q, b in enumerate(init_bits):
        if b:
            init_qc.x(q)
    full = init_qc.compose(fid_qc)
    sv = Statevector(full)
    return np.array([float(np.real(sv.expectation_value(o))) for o in obs_list])


def run():
    print("Phase 1: Training Data Generation")

    print("\nBuilding noise backend ...")
    noisy_sim, ideal_sim, noise_model = create_simulators()
    print(f"Basis gates in noise model : {noise_model.basis_gates}")

    ansatz = build_ansatz(N_QUBITS, N_LAYERS)
    print(
        f"\nAnsatz : {ansatz.num_qubits} qubits | "
        f"{ansatz.num_parameters} params | depth={ansatz.depth()}"
    )

    observables = make_nn_observables(N_QUBITS)
    n_obs = len(observables)
    print(f"Observables : {n_obs} two-qubit Pauli terms on NN pairs")

    g_values = np.linspace(0.4, 2.0, N_G_VALUES)
    J = 1.0
    all_noisy, all_ideal = [], []
    vqe_records, noise_stats = [], {}

    print(
        f"\nGenerating data  ({N_G_VALUES} g-values x {N_PROD_STATES} states) ...\n"
    )
    t0 = time.time()

    for gi, g in enumerate(g_values):
        print(f"  [{gi + 1:2d}/{N_G_VALUES}]  g={g:.3f}", end="  ", flush=True)
        H = make_hamiltonian(N_QUBITS, J, g)
        pv_diag = rng.uniform(-np.pi, np.pi, ansatz.num_parameters)
        bound = ansatz.assign_parameters(dict(zip(ansatz.parameters, pv_diag)))

        ising_obs = [
            SparsePauliOp(t, [c]) for t, c in zip(H.paulis.to_labels(), H.coeffs)
        ]
        noisy_E = sum(expval(bound, o, noisy_sim, SHOTS // 2) for o in ising_obs)
        ideal_E = sum(expval(bound, o, ideal_sim, SHOTS // 2) for o in ising_obs)

        g_noisy, g_ideal = [], []
        for _ in range(N_PROD_STATES):
            pv = rng.uniform(-np.pi, np.pi, ansatz.num_parameters)
            fid_qc = build_fiducial(ansatz, pv)

            init_qc = QuantumCircuit(N_QUBITS)
            angles = rng.uniform(0, 2 * np.pi, (N_QUBITS, 2))
            for q in range(N_QUBITS):
                init_qc.ry(angles[q, 0], q)
                init_qc.rz(angles[q, 1], q)

            full_noisy = init_qc.compose(fid_qc)

            g_noisy.append(expval_vec(full_noisy, observables, noisy_sim, SHOTS))

            sv = Statevector(init_qc.compose(fid_qc))
            ideal_ev = np.array(
                [float(np.real(sv.expectation_value(o))) for o in observables]
            )
            g_ideal.append(ideal_ev)

        g_noisy = np.array(g_noisy)
        g_ideal = np.array(g_ideal)
        all_noisy.append(g_noisy)
        all_ideal.append(g_ideal)

        mae = float(np.mean(np.abs(g_noisy - g_ideal)))
        rmse = float(np.sqrt(np.mean((g_noisy - g_ideal) ** 2)))
        noise_stats[f"g_{g:.3f}"] = {
            "noisy_energy": float(noisy_E),
            "ideal_energy": float(ideal_E),
            "energy_error": float(abs(noisy_E - ideal_E)),
            "fiducial_MAE": mae,
            "fiducial_RMSE": rmse,
        }
        vqe_records.append(
            {"g": float(g), "noisy_E": float(noisy_E), "ideal_E": float(ideal_E)}
        )
        print(
            f"noisy_E={noisy_E:+.4f}  ideal_E={ideal_E:+.4f}  "
            f"fid_MAE={mae:.4f}  fid_RMSE={rmse:.4f}"
        )

    elapsed = time.time() - t0
    print(f"\n  Wall time: {elapsed:.1f} s")

    X_train = np.vstack(all_noisy)
    Y_train = np.vstack(all_ideal)

    print("\nMLP Training Dataset:")
    print(f"    X_train (noisy)  : {X_train.shape}")
    print(f"    Y_train (ideal)  : {Y_train.shape}")
    print(f"    Global MAE       : {np.mean(np.abs(X_train - Y_train)):.4f}")

    np.save(OUTPUT_DIR / "X_train_noisy.npy", X_train)
    np.save(OUTPUT_DIR / "Y_train_ideal.npy", Y_train)

    summary = {
        "description": "DAEM VQE - 4-qubit transverse Ising chain",
        "n_qubits": N_QUBITS,
        "n_layers": N_LAYERS,
        "J": J,
        "g_values": g_values.tolist(),
        "shots_per_obs": SHOTS,
        "n_product_states_per_g": N_PROD_STATES,
        "n_observables": n_obs,
        "X_train_shape": list(X_train.shape),
        "Y_train_shape": list(Y_train.shape),
        "noise_model_basis_gates": noise_model.basis_gates,
        "global_MAE": float(np.mean(np.abs(X_train - Y_train))),
        "global_RMSE": float(np.sqrt(np.mean((X_train - Y_train) ** 2))),
        "noise_stats_per_g": noise_stats,
        "vqe_energy_per_g": vqe_records,
        "elapsed_seconds": round(elapsed, 1),
    }
    with open(OUTPUT_DIR / "daem_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved training data to {OUTPUT_DIR}/")

    sX = StandardScaler().fit(X_train)
    sY = StandardScaler().fit(Y_train)
    Xs = sX.transform(X_train)
    Ys = sY.transform(Y_train)
    Xtr, Xv, Ytr, Yv = train_test_split(Xs, Ys, test_size=0.2, random_state=SEED)

    mlp = MLPRegressor(
        hidden_layer_sizes=(256, 256, 128),
        activation="relu",
        max_iter=500,
        learning_rate_init=1e-3,
        early_stopping=True,
        n_iter_no_change=20,
        random_state=SEED,
        verbose=False,
    )
    mlp.fit(Xtr, Ytr)
    pred = mlp.predict(Xv)
    r2 = r2_score(Yv, pred)
    mae_sc = mean_absolute_error(Yv, pred)
    pred_orig = sY.inverse_transform(pred)
    Yv_orig = sY.inverse_transform(Yv)
    mae_real = mean_absolute_error(Yv_orig, pred_orig)

    print(f"    Validation R2          : {r2:.4f}")
    print(f"    Validation MAE (scaled): {mae_sc:.4f}")
    print(f"    Validation MAE (real)  : {mae_real:.4f}")
    print(f"    Training iterations    : {mlp.n_iter_}")

    summary["mlp check"] = {
        "val_R2": round(r2, 4),
        "val_MAE_scaled": round(mae_sc, 4),
        "val_MAE_real": round(mae_real, 4),
        "n_iters": mlp.n_iter_,
        "architecture": "256-256-128-ReLU",
    }
    with open(OUTPUT_DIR / "daem_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nTraining final DAEM model on all data ...")
    scaler_X_all = StandardScaler()
    scaler_Y_all = StandardScaler()
    X_all_sc = scaler_X_all.fit_transform(X_train)
    Y_all_sc = scaler_Y_all.fit_transform(Y_train)

    mlp_final = MLPRegressor(
        hidden_layer_sizes=(256, 256, 128),
        activation="relu",
        max_iter=2000,
        learning_rate_init=1e-3,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=50,
        random_state=SEED,
        verbose=False,
    )
    mlp_final.fit(X_all_sc, Y_all_sc)

    model_path = OUTPUT_DIR / "daem_model.joblib"
    joblib.dump(
        {"mlp": mlp_final, "scaler_X": scaler_X_all, "scaler_Y": scaler_Y_all},
        model_path,
    )

    best_loss_final = (
        mlp_final.best_loss_ if mlp_final.best_loss_ is not None else mlp_final.loss_
    )
    print(f"    Epochs: {mlp_final.n_iter_}, Loss: {best_loss_final:.6f}")
    print(f"    Saved to {model_path}")

    summary["final_model"] = {
        "n_iters": mlp_final.n_iter_,
        "best_loss": float(best_loss_final),
        "architecture": "256-256-128-ReLU",
    }
    with open(OUTPUT_DIR / "daem_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nPhase 1 complete. ({elapsed:.0f}s)")


if __name__ == "__main__":
    run()
