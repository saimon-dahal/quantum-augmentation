from itertools import product as iproduct

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

from circuit import VQECircuit


def get_pauli_observables(n_qubits):
    """
    Generate single- and two-qubit Pauli observables.

    Qiskit uses LITTLE-ENDIAN ordering: the rightmost character in a Pauli
    string corresponds to qubit 0. So to put operator P on qubit q, we build
    the list s[q]=P and then REVERSE before joining.
    """
    observables = []
    labels = []
    single = ["X", "Y", "Z"]

    # Single-qubit Paulis
    for q in range(n_qubits):
        for p in single:
            s = ["I"] * n_qubits
            s[q] = p
            # FIX: reverse so qubit 0 is rightmost (Qiskit little-endian)
            pauli_str = "".join(reversed(s))
            observables.append(SparsePauliOp(pauli_str))
            labels.append(f"{p}{q}")

    # Two-qubit Paulis (q1 < q2)
    for q1, q2 in iproduct(range(n_qubits), repeat=2):
        if q1 >= q2:
            continue
        for p1, p2 in iproduct(single, repeat=2):
            s = ["I"] * n_qubits
            s[q1] = p1
            s[q2] = p2
            # FIX: reverse so qubit 0 is rightmost
            pauli_str = "".join(reversed(s))
            observables.append(SparsePauliOp(pauli_str))
            labels.append(f"{p1}{q1}{p2}{q2}")

    return observables, labels


def circuit_for_observable(n_qubits, pauli_str):
    """
    Build basis-rotation circuit for a Pauli string.

    pauli_str is Qiskit little-endian: rightmost char = qubit 0.
    We reverse it so that index qb=0 correctly maps to qubit 0.
    """
    circ = QuantumCircuit(n_qubits)
    # FIX: reverse pauli_str so index 0 = qubit 0
    for qb, p in enumerate(reversed(pauli_str)):
        if p == "X":
            circ.h(qb)
        elif p == "Y":
            circ.sdg(qb)
            circ.h(qb)
        # Z and I need no rotation
    return circ


def expectation_from_counts(counts, pauli_str):
    """
    Compute <P> from measurement counts.

    Qiskit returns bitstrings in big-endian order (leftmost = highest qubit).
    We reverse each bitstring so that index 0 = qubit 0.
    pauli_str is Qiskit little-endian so we also reverse it to align with qb index.
    """
    total = 0.0
    total_shots = 0
    # FIX: reverse pauli_str once here to align with reversed bitstring indexing
    reversed_pauli = list(reversed(pauli_str))

    for bitstr, count in counts.items():
        total_shots += count
        # Reverse bitstring: now b[0] = qubit 0
        b = bitstr[::-1]
        sign = 1.0
        for qb, p in enumerate(reversed_pauli):
            if p != "I":
                if int(b[qb]) == 1:
                    sign *= -1.0
        total += sign * count

    return total / total_shots


def ideal_expectation_statevector(circuit, obs):
    """Compute exact <P> via Statevector. Accepts a SparsePauliOp."""
    sv = Statevector(circuit)
    return float(sv.expectation_value(obs).real)


# training_data.py


def compute_expectation_pair(
    theta, n_qubits, n_layers, obs_list, noisy_sim, shots=4096, rng=None
):
    if rng is None:
        rng = np.random.default_rng()

    vqe = VQECircuit(n_qubits, n_layers, theta)
    fiducial = vqe.build_fiducial()

    # FIX: prepend a random state preparation so inputs are non-trivial
    # Random Ry+Rz on each qubit → spreads expvals across [-1, 1]
    state_prep = QuantumCircuit(n_qubits)
    for q in range(n_qubits):
        state_prep.ry(rng.uniform(0, np.pi), q)
        state_prep.rz(rng.uniform(0, 2 * np.pi), q)

    # Full circuit: state_prep → fiducial
    full_circuit = state_prep.compose(fiducial)

    noisy_vec = []
    ideal_vec = []

    for obs in obs_list:
        pauli_str = obs.paulis[0].to_label()

        # Noisy measurement
        rot_circ = circuit_for_observable(n_qubits, pauli_str)
        circ_full = full_circuit.copy()
        circ_full.compose(rot_circ, inplace=True)
        circ_full.measure_all()
        circ_t = transpile(circ_full, noisy_sim, optimization_level=0)
        counts = noisy_sim.run(circ_t, shots=shots).result().get_counts()
        noisy_vec.append(expectation_from_counts(counts, pauli_str))

        # Ideal via Statevector — on the SAME full_circuit (state_prep + fiducial)
        ideal_vec.append(ideal_expectation_statevector(full_circuit, obs))

    return noisy_vec, ideal_vec


def compute_noisy_expvals(circuit, obs_list, noisy_sim, shots=4096):
    """
    Measure a bound (parameter-free) circuit on the noisy simulator.
    Used at inference time on the VQE target circuit.
    """
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


def build_training_dataset(
    n_qubits, n_layers, n_samples, obs_list, noisy_sim, shots=4096, seed=42
):
    rng = np.random.default_rng(seed)
    n_params = 4 * n_qubits * n_layers
    X_rows, y_rows = [], []

    for i in range(n_samples):
        theta = rng.uniform(0, 2 * np.pi, size=n_params)
        print(f"  Sample {i + 1}/{n_samples} ...", end=" ", flush=True)
        noisy_vec, ideal_vec = compute_expectation_pair(
            theta,
            n_qubits,
            n_layers,
            obs_list,
            noisy_sim,
            shots=shots,
            rng=rng,  # ← pass rng so each sample gets different state_prep
        )
        X_rows.append(noisy_vec)
        y_rows.append(ideal_vec)
        print("done")

    return np.array(X_rows), np.array(y_rows)


def make_noisy_sim(fake_backend):
    """Build an AerSimulator with the noise model from a fake backend."""
    noise_model = NoiseModel.from_backend(fake_backend)
    return AerSimulator(noise_model=noise_model)
