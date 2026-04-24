from itertools import product as iproduct

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

from circuit import VQECircuit


def get_pauli_observables(n_qubits):
    """Return single- and two-qubit Pauli observables with human-readable labels.

    Qiskit uses little-endian ordering (rightmost character = qubit 0).
    """
    observables, labels = [], []
    single = ["X", "Y", "Z"]

    for q in range(n_qubits):
        for p in single:
            s = ["I"] * n_qubits
            s[q] = p
            pauli_str = "".join(reversed(s))
            observables.append(SparsePauliOp(pauli_str))
            labels.append(f"{p}{q}")

    for q1, q2 in iproduct(range(n_qubits), repeat=2):
        if q1 >= q2:
            continue
        for p1, p2 in iproduct(single, repeat=2):
            s = ["I"] * n_qubits
            s[q1] = p1
            s[q2] = p2
            pauli_str = "".join(reversed(s))
            observables.append(SparsePauliOp(pauli_str))
            labels.append(f"{p1}{q1}{p2}{q2}")

    return observables, labels

def circuit_for_observable(n_qubits, pauli_str):
    """Build a basis-rotation circuit for measuring *pauli_str*."""
    circ = QuantumCircuit(n_qubits)
    for qb, p in enumerate(reversed(pauli_str)):
        if p == "X":
            circ.h(qb)
        elif p == "Y":
            circ.sdg(qb)
            circ.h(qb)
    return circ


def expectation_from_counts(counts, pauli_str):
    """Compute ⟨P⟩ from measurement *counts* for a given Pauli string."""
    total = 0.0
    total_shots = 0
    reversed_pauli = list(reversed(pauli_str))

    for bitstr, count in counts.items():
        total_shots += count
        b = bitstr[::-1]
        sign = 1.0
        for qb, p in enumerate(reversed_pauli):
            if p != "I" and int(b[qb]) == 1:
                sign *= -1.0
        total += sign * count

    return total / total_shots


def ideal_expectation_statevector(circuit, obs):
    """Compute exact ⟨P⟩ via statevector simulation."""
    sv = Statevector(circuit)
    return float(sv.expectation_value(obs).real)


def compute_noisy_expvals(circuit, obs_list, noisy_sim, shots=4096):
    """Measure all observables on *noisy_sim* and return the expectation-value vector."""
    noisy_vec = []
    for obs in obs_list:
        pauli_str = obs.paulis[0].to_label()
        rot_circ = circuit_for_observable(len(pauli_str), pauli_str)
        circ_full = circuit.copy()
        circ_full.compose(rot_circ, inplace=True)
        circ_full.measure_all()
        circ_t = transpile(circ_full, noisy_sim, optimization_level=0)
        counts = noisy_sim.run(circ_t, shots=shots).result().get_counts()
        noisy_vec.append(expectation_from_counts(counts, pauli_str))
    return np.array(noisy_vec)


def compute_expectation_pair(theta, n_qubits, n_layers, obs_list,
                             noisy_sim, shots=4096, rng=None):
    """Return ``(noisy_vec, ideal_vec)`` for a single fiducial circuit instance.

    A random state-preparation layer (RY + RZ per qubit) is prepended so
    that the resulting expectation values span [−1, +1].
    """
    if rng is None:
        rng = np.random.default_rng()

    vqe = VQECircuit(n_qubits, n_layers, theta)
    fiducial = vqe.build_fiducial()

    state_prep = QuantumCircuit(n_qubits)
    for q in range(n_qubits):
        state_prep.ry(rng.uniform(0, np.pi), q)
        state_prep.rz(rng.uniform(0, 2 * np.pi), q)

    full_circuit = state_prep.compose(fiducial)

    noisy_vec, ideal_vec = [], []
    for obs in obs_list:
        pauli_str = obs.paulis[0].to_label()

        rot_circ = circuit_for_observable(n_qubits, pauli_str)
        circ_full = full_circuit.copy()
        circ_full.compose(rot_circ, inplace=True)
        circ_full.measure_all()
        circ_t = transpile(circ_full, noisy_sim, optimization_level=0)
        counts = noisy_sim.run(circ_t, shots=shots).result().get_counts()
        noisy_vec.append(expectation_from_counts(counts, pauli_str))

        ideal_vec.append(ideal_expectation_statevector(full_circuit, obs))

    return noisy_vec, ideal_vec


def build_training_dataset(n_qubits, n_layers, n_samples, obs_list,
                           noisy_sim, shots=4096, seed=42):
    """Generate ``(X, Y)`` training pairs from fiducial circuits."""
    rng = np.random.default_rng(seed)
    n_params = 4 * n_qubits * n_layers
    X_rows, y_rows = [], []

    for i in range(n_samples):
        theta = rng.uniform(0, 2 * np.pi, size=n_params)
        print(f"Sample {i + 1}/{n_samples}\n")
        noisy_vec, ideal_vec = compute_expectation_pair(
            theta, n_qubits, n_layers, obs_list,
            noisy_sim, shots=shots, rng=rng,
        )
        X_rows.append(noisy_vec)
        y_rows.append(ideal_vec)

    return np.array(X_rows), np.array(y_rows)


def make_noisy_sim(fake_backend):
    """Build an ``AerSimulator`` with the noise model extracted from fake_backend."""
    noise_model = NoiseModel.from_backend(fake_backend)
    return AerSimulator(noise_model=noise_model)
