# VQE + DAEM Error Mitigation

Variational Quantum Eigensolver with Data Augmentation-Empowered Error Mitigation for the transverse-field Ising chain.

## Overview

This project implements a two-phase pipeline that:
1. **Phase 1** — generates paired noisy/ideal measurement data from a simulated quantum backend and trains an MLP regressor to learn the noise map.
2. **Phase 2** — optimizes VQE parameters for each transverse-field strength *g*, then applies the trained MLP to mitigate measurement noise.

The target system is a **4-qubit transverse-field Ising chain**:

```
H = −J Σ ZᵢZᵢ₊₁ − g Σ Xᵢ
```

where *J* = 1.0 (coupling) and *g* is swept across the phase diagram.

## Results

Phase 2 evaluation across four transverse-field strengths (*g*):

| g | Exact Energy | VQE Energy | Noisy Energy | Mitigated Energy | Noisy Error | Mitigated Error |
|---|---|---|---|---|---|---|
| 0.400 | −3.2641 | −3.1225 | −2.9458 | −2.9696 | 0.1922 | 0.1683 |
| 0.933 | −4.5462 | −4.3876 | −4.1735 | −4.0656 | 0.2316 | 0.3396 |
| 1.467 | −6.3821 | −4.6514 | −4.5036 | −4.1565 | 0.1611 | 0.5082 |
| 2.000 | −8.3768 | −8.2737 | −7.9097 | −7.7384 | 0.3911 | 0.5624 |

**Averages**:
- Noisy energy error: **0.2440**
- Mitigated energy error: **0.3946**
- Observable MAE (noisy): 0.0243, Observable MAE (mitigated): 0.0770

The MLP was trained for **223 epochs**. At this noise level (~2% per observable on a simulated backend), mitigation improved the low-*g* regime but did not reduce average error across all field strengths. This is expected — DAEM provides larger gains when hardware noise exceeds the mild levels of the `GenericBackendV2` simulator. 

## Requirements

- Python ≥ 3.12
- [uv](https://docs.astral.sh/uv/) package manager (recommended)

Dependencies (installed automatically):
- `numpy >= 2.4.4`
- `qiskit >= 2.4.0`
- `qiskit-aer >= 0.17.2`
- `scikit-learn >= 1.6.0`

## Setup

```bash
git clone <repository-url>
cd test_factor_quant
uv sync
```

## Usage

Run the phases sequentially — Phase 2 requires the trained model produced by Phase 1.

```bash
# Phase 1: generate training data and train MLP (~20–30 min)
python -m src.phase1

# Phase 2: evaluate error mitigation (~5–15 min)
python -m src.phase2
```

Outputs are written to `p1-outputs/`.

## Configuration

All tunable parameters are in `src/config.py`:

| Parameter | Default | Description |
|---|---|---|
| `N_QUBITS` | 4 | Number of qubits in the Ising chain |
| `N_LAYERS` | 3 | Ansatz circuit depth |
| `SHOTS` | 4096 | Measurement shots per observable |
| `N_G_VALUES` | 16 | Number of field strengths sampled in Phase 1 |
| `N_PROD_STATES` | 100 | Random input states per *g* value |
| `J` | 1.0 | Spin-spin coupling strength |
| `SEED` | 42 | Random seed for reproducibility |

## How It Works

### Phase 1 — Training Data Generation

For each of 16 transverse-field values *g* ∈ [0.4, 2.0]:
1. Construct a **fiducial circuit** from the VQE ansatz where single-qubit rotations are split into cancelling half-rotation pairs (preserving the noise profile while making the ideal output trivially computable).
2. For 100 random input states, measure 45 nearest-neighbour Pauli observables on both noisy and ideal simulators.
3. Train an MLP (256-256-128, ReLU) on the 1600 × 45 noisy→ideal mapping.

### Phase 2 — Error Mitigation

For each evaluation *g* value:
1. Optimize VQE parameters via COBYLA on exact Statevector simulation.
2. Measure the bound circuit on noisy and ideal backends.
3. Pass noisy expectation values through the trained MLP to produce mitigated values.
4. Compare energy errors: `|noisy − ideal|` vs `|mitigated − ideal|`.
