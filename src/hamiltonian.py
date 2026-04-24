from qiskit.quantum_info import SparsePauliOp


def make_hamiltonian(n: int, J: float, g: float) -> SparsePauliOp:
    """Build the transverse Ising Hamiltonian: H = -J sum(ZZ) - g sum(X)."""
    ops, coeffs = [], []

    for i in range(n - 1):
        label = ["I"] * n
        label[i] = "Z"
        label[i + 1] = "Z"
        ops.append("".join(reversed(label)))
        coeffs.append(-J)

    for i in range(n):
        label = ["I"] * n
        label[i] = "X"
        ops.append("".join(reversed(label)))
        coeffs.append(-g)

    return SparsePauliOp(ops, coeffs)
