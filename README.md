# DAEM: Data Augmentation-Empowered Error Mitigation

A minimal implementation of the DAEM technique for mitigating noise in quantum circuits, based on the approach in:

> Manwen Liao, Yan Zhu, Giulio Chiribella, Yuxiang Yang, *Noise-agnostic quantum error mitigation with data augmented neural models*, npj Quantum Information, 2025.

The core idea: train a neural network on (noisy, ideal) measurement pairs from *fiducial circuits* — circuits that share the noise profile of the real workload but whose ideal outputs are cheap to compute classically. The trained model then denoises measurements from the actual quantum circuit of interest.

## Method

1. **Fiducial circuit construction.** Each single-qubit rotation R(θ) in the VQE ansatz is replaced by the identity pair R(−θ/2)·R(+θ/2). The gate structure (and therefore the noise) is preserved, but the ideal expectation values are analytically known via statevector simulation.

2. **Training data generation.** For each of *N* random parameter vectors, a random state-preparation layer is prepended and the fiducial circuit is measured on a noisy simulator. The ideal values are computed exactly with a statevector backend. This yields paired training data (X_noisy, Y_ideal).

3. **Residual MLP.** A small multi-layer perceptron learns the correction ΔX so that X + ΔX ≈ Y. The residual connection biases the network toward learning the noise, not the signal.

4. **Inference.** The target VQE circuit is measured on the noisy simulator, and the trained MLP denoises the resulting expectation-value vector.


## Setup

```bash
# Install dependencies (requires uv)
uv sync

# Run
uv run python main.py
```

Runtime is roughly 4–5 minutes on a modern laptop (200 fiducial samples, 300 MLP epochs).

## Configuration

All hyperparameters are defined at the top of `main.py`:

| Parameter   | Default | Description                      |
|-------------|---------|----------------------------------|
| `N_QUBITS`  | 3       | Number of qubits                 |
| `N_LAYERS`  | 2       | Ansatz depth                     |
| `N_SAMPLES` | 200     | Training samples (fiducial runs) |
| `SHOTS`     | 4096    | Measurement shots per circuit    |
| `EPOCHS`    | 300     | MLP training epochs              |
| `LR`        | 1e-3    | Learning rate (Adam)             |
| `HIDDEN`    | 128     | Hidden layer width               |
| `SEED`      | 42      | Random seed                      |

The noisy backend is `FakeNairobiV2` from `qiskit-ibm-runtime`.

## Results

Configuration: 3 qubits, 2 layers, 200 training samples, 4096 shots, `FakeNairobiV2` noise model.

### Summary

| Metric                   | Value  |
|--------------------------|--------|
| MAE (noisy vs ideal)     | 0.0437 |
| MAE (mitigated vs ideal) | 0.0312 |
| **Error reduction**      | **28.7%** |
| Best validation loss     | 7.6e-4 |

### Per-Observable Breakdown

36 single- and two-qubit Pauli observables are measured. A subset:

| Observable | Noisy    | Mitigated | Ideal    | Err (noisy) | Err (mitigated) |
|------------|----------|-----------|----------|-------------|-----------------|
| Y0         | −0.4707  | −0.5380   | −0.5774  | 0.1067      | 0.0394          |
| X0Y2       | +0.5566  | +0.7431   | +0.6938  | 0.1372      | 0.0493          |
| Z0Z2       | +0.3945  | +0.5320   | +0.5028  | 0.1083      | 0.0292          |
| Y0Z2       | −0.3413  | −0.3888   | −0.4450  | 0.1037      | 0.0562          |
| X0Y1       | −0.3511  | −0.3684   | −0.4471  | 0.0960      | 0.0787          |
| Y1Y2       | −0.6178  | −0.6335   | −0.6996  | 0.0820      | 0.0661          |
| X1X2       | −0.4268  | −0.4451   | −0.5046  | 0.0778      | 0.0595          |
| Z1Z2       | −0.5977  | −0.6737   | −0.6690  | 0.0713      | 0.0047          |

Full per-observable results are saved in `results.json`.

## Dependencies

- Python ≥ 3.12
- Qiskit ≥ 2.3, Qiskit Aer ≥ 0.17, qiskit-ibm-runtime ≥ 0.46
- PyTorch ≥ 2.11
- NumPy, Matplotlib, scikit-learn
