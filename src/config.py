from pathlib import Path

import numpy as np

N_QUBITS = 4
N_LAYERS = 3
SHOTS = 4096
N_G_VALUES = 16
N_PROD_STATES = 100
J = 1.0
SEED = 42
PAULI_CHARS = ["I", "X", "Y", "Z"]

OUTPUT_DIR = Path("p1-outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(SEED)
