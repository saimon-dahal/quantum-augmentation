from itertools import product as iproduct

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer.primitives import Sampler

from circuit import VQECircuit


def get_pauli_observables(n_qubits):
    """Generate single- and two-qubit Paulis"""
    observables = []
    labels = []
    single = ["X", "Y", "Z"]

    # Single‑qubit Paulis
    for q in range(n_qubits):
        for p in single:
            s = ["I"] * n_qubits
            s[q] = p
            pauli_str = "".join(s)
            observables.append(SparsePauliOp(pauli_str))
            labels.append(f"{p}{q}")

    # Two‑qubit Paulis (q1 < q2 to avoid duplication)
    for q1, q2 in iproduct(range(n_qubits), repeat=2):
        if q1 >= q2:
            continue
        for p1, p2 in iproduct(single, repeat=2):
            s = ["I"] * n_qubits
            s[q1] = p1
            s[q2] = p2
            pauli_str = "".join(s)
            observables.append(SparsePauliOp(pauli_str))
            labels.append(f"{p1}{q1}{p2}{q2}")

    return observables, labels


def circuit_for_observable(n_qubits, pauli_str):
    """
    Build rotation circuit for a Pauli string (e.g., "ZIZ", "XXI").
    qb=0 is leftmost qubit in the string.
    """
    circ = QuantumCircuit(n_qubits)
    for qb, p in enumerate(pauli_str):
        if p == "X":
            circ.h(qb)
        elif p == "Y":
            circ.sdg(qb)
            circ.h(qb)
        # Z needs no rotation
    return circ


def expectation_from_counts(counts, pauli_str):
    """
    Compute <P> from measurement counts.
    """
    total = 0.0
    total_shots = 0

    for bitstr, count in counts.items():
        total_shots += count
        # Reverse the bitstring so index 0 = qubit 0
        b = bitstr[::-1]  # <-- THE FIX

        sign = 1.0
        for qb, p in enumerate(pauli_str):
            if p != "I":
                val = int(b[qb])  # now b[0] = qubit 0
                if val == 1:
                    sign *= -1.0
        total += sign * count

    return total / total_shots


def ideal_expectation_statevector(circuit, pauli_str):
    sv = Statevector(circuit)
    op = SparsePauliOp(pauli_str)
    return float(sv.expectation_value(op).real)


def compute_expectation_pair(
    theta, n_qubits, n_layers, obs_list, obs_labels, noisy_backend, shots=4096
):
    """
    For a single theta, compute:
        noisy_vec : list of noisy <P_i> values (from FakeNairobi)
        ideal_vec : list of exact  <P_i> values (from Statevector)
    """
    vqe = VQECircuit(n_qubits, n_layers, theta)
    fiducial = vqe.build_fiducial()

    noisy_sampler = Sampler(backend_options=noisy_backend)

    noisy_vec = []
    ideal_vec = []

    for op, lbl in zip(obs_list, obs_labels):
        pauli_str = op.paulis[0].to_label()

        rot_circ = circuit_for_observable(n_qubits, pauli_str)
        circ_full = fiducial.copy()
        circ_full.compose(rot_circ, inplace=True)
        circ_full.measure_all()
        circ_t = transpile(circ_full, noisy_backend)

        job = noisy_sampler.run([circ_t], shots=shots)
        counts = job.result()[0].data.meas.get_counts()
        noisy_val = expectation_from_counts(counts, pauli_str)
        noisy_vec.append(noisy_val)

        ideal_val = ideal_expectation_statevector(fiducial, pauli_str)
        ideal_vec.append(ideal_val)

    return noisy_vec, ideal_vec


def build_training_dataset(
    n_qubits,
    n_layers,
    n_samples,
    obs_list,
    obs_labels,
    noisy_backend,
    shots=4096,
    seed=42,
):
    """
    Build the full training dataset.
    """
    rng = np.random.default_rng(seed)
    n_params = 4 * n_qubits * n_layers

    X_rows = []
    y_rows = []

    for i in range(n_samples):
        # Sample a random parameter vector uniformly in [0, 2π]
        theta = rng.uniform(0, 2 * np.pi, size=n_params)

        print(f"Sample {i + 1}/{n_samples} ...", end=" ", flush=True)
        noisy_vec, ideal_vec = compute_expectation_pair(
            theta, n_qubits, n_layers, obs_list, obs_labels, noisy_backend, shots=shots
        )
        X_rows.append(noisy_vec)
        y_rows.append(ideal_vec)
        print("done")

    X = np.array(X_rows)
    y = np.array(y_rows)
    return X, y


def build_state_prep(n_qubits, state_label):
    """
    Prepend a state-preparation unitary before the fiducial.
    """
    qc = QuantumCircuit(n_qubits)
    if state_label == "0":
        pass
    elif state_label == "1":
        for q in range(n_qubits):
            qc.x(q)
    elif state_label == "+":
        for q in range(n_qubits):
            qc.h(q)
    elif state_label == "random":
        rng = np.random.default_rng()
        for q in range(n_qubits):
            qc.rx(rng.uniform(0, np.pi), q)
            qc.ry(rng.uniform(0, np.pi), q)
    return qc
