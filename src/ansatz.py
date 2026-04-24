from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


def build_ansatz(n: int, depth: int) -> QuantumCircuit:
    """Build a hardware-efficient VQE ansatz with Euler (RZ-RY-RZ) rotations and CNOT ladder."""
    params = ParameterVector("th", 3 * n * (depth + 1))
    qc = QuantumCircuit(n)
    idx = 0

    for layer in range(depth + 1):
        for q in range(n):
            qc.rz(params[idx], q)
            qc.ry(params[idx + 1], q)
            qc.rz(params[idx + 2], q)
            idx += 3
        if layer < depth:
            for q in range(n - 1):
                qc.cx(q, q + 1)

    return qc


def build_fiducial(ansatz_qc: QuantumCircuit, param_vals) -> QuantumCircuit:
    """Build a fiducial circuit by splitting single-qubit rotations into identity pairs."""
    bound = ansatz_qc.assign_parameters(dict(zip(ansatz_qc.parameters, param_vals)))
    fid_qc = QuantumCircuit(ansatz_qc.num_qubits)

    for instr in bound.data:
        gate = instr.operation
        qargs = [fid_qc.qubits[bound.find_bit(q).index] for q in instr.qubits]
        if gate.num_qubits == 1 and gate.name in ("rz", "ry", "rx"):
            phi = float(gate.params[0])
            getattr(fid_qc, gate.name)(-phi / 2, qargs[0])
            getattr(fid_qc, gate.name)(phi / 2, qargs[0])
        else:
            fid_qc.append(gate, qargs)

    return fid_qc
