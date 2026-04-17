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
        """Build hardware-aware fiducial circuit (identity ideally)"""
        qc = QuantumCircuit(self.n_qubits)
        idx = 0
        for _ in range(self.n_layers):
            for q in range(self.n_qubits):
                # Identity-equivalent Rx
                qc.rx(-self.theta[idx], q)
                qc.rx(self.theta[idx], q)
                idx += 1
                # Identity-equivalent Ry
                qc.ry(-self.theta[idx], q)
                qc.ry(self.theta[idx], q)
                idx += 1

            for q in range(self.n_qubits - 1):
                qc.cx(q, q + 1)

            for q in range(self.n_qubits):
                # Identity-equivalent Ry
                qc.ry(-self.theta[idx], q)
                qc.ry(self.theta[idx], q)
                idx += 1
                # Identity-equivalent Rz
                qc.rz(-self.theta[idx], q)
                qc.rz(self.theta[idx], q)
                idx += 1
        return qc
    def build_fiducial_ideal(self):
        """
        The ideal version for classical label computation.
        Single-qubit gates are exactly identity — only CNOTs remain.
        This IS Clifford and classically simulable in O(N).
        """
        qc = QuantumCircuit(self.n_qubits)
        for _ in range(self.n_layers):
            # Single-qubit gates → identity (omitted entirely)
            for q in range(self.n_qubits - 1):
                qc.cx(q, q + 1)   # CNOTs preserved
        return qc

from qiskit.quantum_info import Clifford

def is_clifford(qc):
    try:
        Clifford(qc)
        return True
    except Exception as e:
        print(f"Not Clifford: {e}")
        return False

import numpy as np
# Test
vqe = VQECircuit(n_qubits=2, n_layers=1, theta=np.random.rand(12))
print(is_clifford(vqe.build_fiducial()))   # Should be False (arbitrary θ)
print(is_clifford(vqe.build_target()))     # Also False
