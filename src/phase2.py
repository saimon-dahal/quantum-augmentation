"""
Phase 2: DAEM Error Mitigation
===============================
Loads the trained MLP from Phase 1, optimizes VQE parameters for each
transverse-field strength g, then compares noisy vs. MLP-mitigated
expectation values against ideal results.

Outputs (to OUTPUT_DIR):
    daem_phase2_results.json — per-g results and aggregate statistics
"""

import json
import warnings

import joblib
import numpy as np

from .ansatz import build_ansatz
from .backend import create_simulators
from .config import J, N_LAYERS, N_QUBITS, OUTPUT_DIR, SHOTS, rng
from .hamiltonian import make_hamiltonian
from .measurement import energy_from_expvals, expval_vec
from .observables import make_nn_observables
from .vqe import exact_ground_energy, optimize_vqe

warnings.filterwarnings("ignore")

G_VALUES = np.linspace(0.4, 2.0, 4)


def run():
    print("\nSetting up simulators...")
    noisy_sim, ideal_sim, noise_model = create_simulators()
    print(f"   Noise model gates: {noise_model.basis_gates}")

    ansatz = build_ansatz(N_QUBITS, N_LAYERS)
    print(
        f"\nAnsatz: {ansatz.num_qubits} qubits, "
        f"{ansatz.num_parameters} params, depth={ansatz.depth()}"
    )

    observables = make_nn_observables(N_QUBITS)
    n_obs = len(observables)
    print(f"Observables: {n_obs} NN Pauli terms")

    print("\nLoading trained DAEM model...")
    model_path = OUTPUT_DIR / "daem_model.joblib"
    model_data = joblib.load(model_path)
    mlp = model_data["mlp"]
    scaler_X = model_data["scaler_X"]
    scaler_Y = model_data["scaler_Y"]
    print(f"   Loaded from {model_path}")
    print(f"   MLP epochs at save: {mlp.n_iter_}")

    results = []

    for g_idx, g in enumerate(G_VALUES):
        print(f"\n[{g_idx + 1}/{len(G_VALUES)}] g = {g:.3f}")

        H = make_hamiltonian(N_QUBITS, J, g)

        exact_E = exact_ground_energy(H)
        print(f"   Exact ground energy : {exact_E:+.4f}")

        print("   Optimizing VQE parameters ...")
        opt_params, vqe_E = optimize_vqe(ansatz, H, rng, maxiter=300)
        print(f"   VQE optimized energy: {vqe_E:+.4f}  (gap: {abs(vqe_E - exact_E):.4f})")

        bound_circuit = ansatz.assign_parameters(
            dict(zip(ansatz.parameters, opt_params))
        )

        print("   Measuring noisy circuit ...")
        noisy_expvals = expval_vec(bound_circuit, observables, noisy_sim, shots=SHOTS)
        noisy_energy = energy_from_expvals(noisy_expvals, observables, H)
        print(f"   Noisy energy:     {noisy_energy:+.4f}")

        print("   Measuring ideal circuit ...")
        ideal_expvals = expval_vec(bound_circuit, observables, ideal_sim, shots=SHOTS)
        ideal_energy = energy_from_expvals(ideal_expvals, observables, H)
        print(f"   Ideal energy:     {ideal_energy:+.4f}")

        print("   Applying MLP mitigation ...")
        noisy_scaled = scaler_X.transform(noisy_expvals.reshape(1, -1))
        mitigated_scaled = mlp.predict(noisy_scaled)
        mitigated_expvals = scaler_Y.inverse_transform(mitigated_scaled).flatten()
        mitigated_energy = energy_from_expvals(mitigated_expvals, observables, H)
        print(f"   Mitigated energy: {mitigated_energy:+.4f}")

        noisy_error = abs(noisy_energy - ideal_energy)
        mitigated_error = abs(mitigated_energy - ideal_energy)
        improvement = noisy_error - mitigated_error

        obs_mae_noisy = float(np.mean(np.abs(noisy_expvals - ideal_expvals)))
        obs_mae_mitigated = float(np.mean(np.abs(mitigated_expvals - ideal_expvals)))

        print(f"   Noisy error:     {noisy_error:.4f}")
        print(f"   Mitigated error: {mitigated_error:.4f}")
        print(f"   Improvement:     {improvement:+.4f}")

        results.append({
            "g": float(g),
            "exact_ground_energy": float(exact_E),
            "vqe_optimized_energy": float(vqe_E),
            "ideal_energy": float(ideal_energy),
            "noisy_energy": float(noisy_energy),
            "mitigated_energy": float(mitigated_energy),
            "noisy_error": float(noisy_error),
            "mitigated_error": float(mitigated_error),
            "improvement": float(improvement),
            "obs_mae_noisy": float(obs_mae_noisy),
            "obs_mae_mitigated": float(obs_mae_mitigated),
        })

    print("\nSummary")

    avg_noisy_err = np.mean([r["noisy_error"] for r in results])
    avg_mitigated_err = np.mean([r["mitigated_error"] for r in results])
    avg_obs_noisy = np.mean([r["obs_mae_noisy"] for r in results])
    avg_obs_mitigated = np.mean([r["obs_mae_mitigated"] for r in results])

    print(
        f"\n  {'g':>6} | {'Exact E':>10} | {'Ideal E':>10} | "
        f"{'Noisy E':>10} | {'Mitig E':>12} | {'N err':>8} | {'M err':>8}"
    )
    print("  " + "-" * 80)
    for r in results:
        print(
            f"  {r['g']:6.3f} | {r['exact_ground_energy']:+10.4f} | "
            f"{r['ideal_energy']:+10.4f} | {r['noisy_energy']:+10.4f} | "
            f"{r['mitigated_energy']:+12.4f} | {r['noisy_error']:8.4f} | "
            f"{r['mitigated_error']:8.4f}"
        )

    print(f"\n  Avg energy error (noisy)    : {avg_noisy_err:.4f}")
    print(f"  Avg energy error (mitigated): {avg_mitigated_err:.4f}")
    print(f"\n  Avg observable MAE (noisy)     : {avg_obs_noisy:.4f}")
    print(f"  Avg observable MAE (mitigated) : {avg_obs_mitigated:.4f}")

    if avg_mitigated_err < avg_noisy_err:
        reduction = 100 * (avg_noisy_err - avg_mitigated_err) / avg_noisy_err
        print(f"\n  Energy error reduced by {reduction:.1f}%")
    else:
        print("\n  MLP mitigation did not reduce average energy error.")

    phase2_out = {
        "description": "DAEM Phase 2 — error mitigation with optimized VQE",
        "n_qubits": N_QUBITS,
        "J": J,
        "g_values": G_VALUES.tolist(),
        "per_g_results": results,
        "avg_noisy_energy_error": float(avg_noisy_err),
        "avg_mitigated_energy_error": float(avg_mitigated_err),
        "avg_obs_mae_noisy": float(avg_obs_noisy),
        "avg_obs_mae_mitigated": float(avg_obs_mitigated),
        "mlp_epochs": int(mlp.n_iter_),
    }
    with open(OUTPUT_DIR / "daem_phase2_results.json", "w") as f:
        json.dump(phase2_out, f, indent=2)

    print(f"\n  Results saved to {OUTPUT_DIR}/daem_phase2_results.json")
    print("  Phase 2 complete.")


if __name__ == "__main__":
    run()
