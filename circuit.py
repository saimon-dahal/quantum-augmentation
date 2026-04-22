from qiskit import QuantumCircuit


class VQECircuit:
    """
    Class for constructing both target VQE and hardware-aware fiducial circuits.
    """

    def __init__(self, n_qubits, n_layers, theta):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.theta = theta  # full parameter array

    def build_target(self):
        """Build the target VQE circuit"""
        qc = QuantumCircuit(self.n_qubits)
        idx = 0
        for _ in range(self.n_layers):
            for q in range(self.n_qubits):
                qc.rx(self.theta[idx], q)
                idx += 1
                qc.ry(self.theta[idx], q)
                idx += 1
            for q in range(self.n_qubits - 1):
                qc.cx(q, q + 1)
            for q in range(self.n_qubits):
                qc.ry(self.theta[idx], q)
                idx += 1
                qc.rz(self.theta[idx], q)
                idx += 1
        return qc

    def build_fiducial(self):
        """
        Build hardware-aware fiducial circuit.
        Each single-qubit gate R(theta) is split into R(-theta/2) R(+theta/2),
        which is identity under noiseless conditions but exposes the same
        gate structure (and thus similar noise) as the target circuit.
        CNOTs are kept unchanged.
        """
        qc = QuantumCircuit(self.n_qubits)
        idx = 0
        for _ in range(self.n_layers):
            for q in range(self.n_qubits):
                # FIX: use half-angles so noise profile matches target
                qc.rx(-self.theta[idx] / 2, q)
                qc.rx(self.theta[idx] / 2, q)
                idx += 1
                qc.ry(-self.theta[idx] / 2, q)
                qc.ry(self.theta[idx] / 2, q)
                idx += 1
            for q in range(self.n_qubits - 1):
                qc.cx(q, q + 1)
            for q in range(self.n_qubits):
                qc.ry(-self.theta[idx] / 2, q)
                qc.ry(self.theta[idx] / 2, q)
                idx += 1
                qc.rz(-self.theta[idx] / 2, q)
                qc.rz(self.theta[idx] / 2, q)
                idx += 1
        return qc
