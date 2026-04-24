from itertools import product as iproduct

from qiskit.quantum_info import SparsePauliOp

from .config import PAULI_CHARS


def make_nn_observables(n: int) -> list[SparsePauliOp]:
    """Create all non-trivial two-qubit Pauli observables on nearest-neighbour pairs."""
    obs_list = []
    for i in range(n - 1):
        for p1, p2 in iproduct(PAULI_CHARS, PAULI_CHARS):
            if p1 == "I" and p2 == "I":
                continue
            label = ["I"] * n
            label[i] = p1
            label[i + 1] = p2
            obs_list.append(SparsePauliOp("".join(reversed(label))))
    return obs_list
