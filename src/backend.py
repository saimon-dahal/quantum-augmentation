from qiskit.providers.fake_provider import GenericBackendV2
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

from .config import N_QUBITS, SEED


def create_simulators():
    """Create noisy and ideal AerSimulators from a fake backend."""
    fake_backend = GenericBackendV2(
        num_qubits=N_QUBITS,
        basis_gates=["cx", "u3", "id", "reset", "measure"],
        seed=SEED,
    )
    noise_model = NoiseModel.from_backend(fake_backend)
    noisy_sim = AerSimulator(noise_model=noise_model)
    ideal_sim = AerSimulator()
    return noisy_sim, ideal_sim, noise_model
