import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator

from .config import N_QUBITS
from .hamiltonian import make_hamiltonian


def expval(
    circuit: QuantumCircuit,
    observable: SparsePauliOp,
    simulator: AerSimulator,
    shots: int,
) -> float:
    """Estimate the expectation value of a single Pauli observable via shot-based sampling."""
    pauli_label = observable.paulis.to_labels()[0]

    meas_circuit = circuit.copy()
    meas_circuit.barrier()
    for qi, pc in enumerate(reversed(pauli_label)):
        if pc == "X":
            meas_circuit.h(qi)
        elif pc == "Y":
            meas_circuit.sdg(qi)
            meas_circuit.h(qi)
    meas_circuit.measure_all()

    transpiled = transpile(meas_circuit, simulator, optimization_level=0)
    counts = simulator.run(transpiled, shots=shots).result().get_counts()

    ev = 0.0
    for bitstring, count in counts.items():
        bitstring = bitstring.replace(" ", "")
        parity = sum(
            int(bitstring[-(qi + 1)])
            for qi, pc in enumerate(reversed(pauli_label))
            if pc != "I"
        )
        ev += (-1) ** parity * count

    coeff = float(np.real(observable.coeffs[0]))
    return coeff * ev / shots


def expval_vec(
    circuit: QuantumCircuit,
    obs_list: list[SparsePauliOp],
    simulator: AerSimulator,
    shots: int,
) -> np.ndarray:
    """Measure all observables and return results as a numpy array."""
    return np.array([expval(circuit, o, simulator, shots) for o in obs_list])


def compute_energy(
    circuit: QuantumCircuit,
    g: float,
    J: float,
    simulator: AerSimulator,
    shots: int,
) -> float:
    """Compute the Ising Hamiltonian energy for a bound circuit."""
    H = make_hamiltonian(N_QUBITS, J, g)
    energy = 0.0
    for pauli_str, coeff in zip(H.paulis.to_labels(), H.coeffs):
        obs = SparsePauliOp(pauli_str, [coeff])
        energy += expval(circuit, obs, simulator, shots)
    return energy


def energy_from_expvals(
    expvals: np.ndarray,
    observables: list[SparsePauliOp],
    hamiltonian: SparsePauliOp,
) -> float:
    """Compute Hamiltonian energy from pre-measured observable expectation values."""
    obs_labels = [o.paulis.to_labels()[0] for o in observables]
    energy = 0.0
    for pauli_str, coeff in zip(hamiltonian.paulis.to_labels(), hamiltonian.coeffs):
        if pauli_str in obs_labels:
            idx = obs_labels.index(pauli_str)
            energy += float(np.real(coeff)) * expvals[idx]
    return energy
