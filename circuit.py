from qiskit import QuantumCircuit


class VQECircuit:
    """Constructs target VQE and hardware-aware fiducial circuits."""

    def __init__(self, n_qubits: int, n_layers: int, theta):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.theta = theta

    def build_target(self) -> QuantumCircuit:
        """Return the target VQE ansatz bound with ``self.theta``."""
        qc = QuantumCircuit(self.n_qubits)
        idx = 0
        for _ in range(self.n_layers):
            for q in range(self.n_qubits):
                qc.rx(self.theta[idx], q); idx += 1
                qc.ry(self.theta[idx], q); idx += 1
            for q in range(self.n_qubits - 1):
                qc.cx(q, q + 1)
            for q in range(self.n_qubits):
                qc.ry(self.theta[idx], q); idx += 1
                qc.rz(self.theta[idx], q); idx += 1
        return qc

    def build_fiducial(self) -> QuantumCircuit:
        """Return the fiducial circuit corresponding to ``self.theta``.

        Single-qubit rotations are split into cancelling half-angle pairs
        (identity under noiseless execution).  CNOTs are kept unchanged.
        """
        qc = QuantumCircuit(self.n_qubits)
        idx = 0
        for _ in range(self.n_layers):
            for q in range(self.n_qubits):
                qc.rx(-self.theta[idx] / 2, q)
                qc.rx(+self.theta[idx] / 2, q); idx += 1
                qc.ry(-self.theta[idx] / 2, q)
                qc.ry(+self.theta[idx] / 2, q); idx += 1
            for q in range(self.n_qubits - 1):
                qc.cx(q, q + 1)
            for q in range(self.n_qubits):
                qc.ry(-self.theta[idx] / 2, q)
                qc.ry(+self.theta[idx] / 2, q); idx += 1
                qc.rz(-self.theta[idx] / 2, q)
                qc.rz(+self.theta[idx] / 2, q); idx += 1
        return qc
