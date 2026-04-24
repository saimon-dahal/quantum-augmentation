"""VQE parameter optimization using exact Statevector simulation."""

import numpy as np
from scipy.optimize import minimize
from qiskit.quantum_info import SparsePauliOp, Statevector


def vqe_cost(params, ansatz, hamiltonian):
    """Compute exact energy for given parameters via Statevector."""
    bound = ansatz.assign_parameters(dict(zip(ansatz.parameters, params)))
    sv = Statevector(bound)
    return float(np.real(sv.expectation_value(hamiltonian)))


def optimize_vqe(ansatz, hamiltonian, rng, maxiter=300):
    """Find optimal VQE parameters using COBYLA optimizer.

    Returns (optimal_params, optimal_energy).
    """
    x0 = rng.uniform(-np.pi, np.pi, ansatz.num_parameters)
    result = minimize(
        vqe_cost,
        x0,
        args=(ansatz, hamiltonian),
        method="COBYLA",
        options={"maxiter": maxiter, "rhobeg": 0.5},
    )
    return result.x, result.fun


def exact_ground_energy(hamiltonian: SparsePauliOp) -> float:
    """Compute exact ground state energy via full diagonalization."""
    mat = hamiltonian.to_matrix()
    if hasattr(mat, "toarray"):
        mat = mat.toarray()
    return float(np.linalg.eigvalsh(mat)[0])
