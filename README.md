# hyper-surrogate

[![Release](https://img.shields.io/github/v/release/jpsferreira/hyper-surrogate)](https://img.shields.io/github/v/release/jpsferreira/hyper-surrogate)
[![Build status](https://img.shields.io/github/actions/workflow/status/jpsferreira/hyper-surrogate/main.yml?branch=main)](https://github.com/jpsferreira/hyper-surrogate/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/jpsferreira/hyper-surrogate/branch/main/graph/badge.svg)](https://codecov.io/gh/jpsferreira/hyper-surrogate)
[![License](https://img.shields.io/github/license/jpsferreira/hyper-surrogate)](https://img.shields.io/github/license/jpsferreira/hyper-surrogate)

Data-driven surrogates for hyperelastic constitutive models in finite element analysis.

- **Github repository**: <https://github.com/jpsferreira/hyper-surrogate/>
- **Documentation**: <https://jpsferreira.github.io/hyper-surrogate/>

## Table of Contents

- [Why hyper-surrogate?](#why-hyper-surrogate)
- [Architecture](#architecture)
- [Features](#features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Examples](#examples)
- [Contributing](#contributing)

## Why hyper-surrogate?

Finite element solvers need constitutive models (stress-strain relationships) to simulate materials. Writing and maintaining these models — especially user-defined material subroutines (UMATs) for Abaqus/LS-DYNA — is tedious, error-prone, and tightly coupled to the solver.

**hyper-surrogate** provides an end-to-end pipeline that:

1. **Generates training data** from symbolic material definitions (NeoHooke, Mooney-Rivlin, etc.)
2. **Trains neural network surrogates** (MLP or Input-Convex NN) that learn the strain energy function
3. **Exports to standalone Fortran 90** with baked-in weights — no Python or external dependencies at runtime

This gives you **solver portability** (one training, any Fortran-capable solver), **thermodynamic consistency** (energy-based formulation with automatic stress/tangent derivation), and **speed** (the exported Fortran is a pure forward pass).

## Architecture

```
Material ─> DeformationGenerator ─> Dataset ─> MLP / ICNN ─> FortranEmitter ─> .f90
   │                                              │
   │         (symbolic SEF)                        │  (trained NN weights)
   │                                              │
   └──── UMATHandler ─────────────────────────────┘
         (analytical Fortran via SymPy CSE)     HybridUMATEmitter
                                                (NN SEF + analytical mechanics)
```

## Features

- **Symbolic mechanics** -- Automatic PK2 stress and stiffness tensor derivation via SymPy
- **ML surrogates** -- MLP and Input-Convex Neural Network (ICNN) models for constitutive response approximation
- **Hybrid UMAT** -- NN-based strain energy with analytical kinematics, stress push-forward, and tangent computation
- **Energy-stress loss** -- Joint energy + stress gradient loss for thermodynamic consistency via autograd
- **ICNN convexity** -- Input-Convex Neural Networks guarantee convexity of the predicted strain energy
- **Fortran export** -- Transpile trained models to standalone Fortran 90 subroutines with baked-in weights
- **Analytical UMAT** -- Generate optimized Fortran UMAT subroutines from symbolic expressions via CSE

## Quick Start

### MLP surrogate (stress-based)

```python
import hyper_surrogate as hs

# Define material and generate training data
material = hs.NeoHooke({"C10": 0.5, "KBULK": 1000.0})
train_ds, val_ds, in_norm, out_norm = hs.create_datasets(
    material, n_samples=5000, input_type="invariants", target_type="pk2_voigt",
)

# Train an MLP surrogate
model = hs.MLP(input_dim=3, output_dim=6, hidden_dims=[32, 32], activation="tanh")
result = hs.Trainer(model, train_ds, val_ds, loss_fn=hs.StressLoss(), max_epochs=500).fit()

# Export to Fortran 90
exported = hs.extract_weights(result.model, in_norm, out_norm)
hs.FortranEmitter(exported).write("nn_surrogate.f90")
```

### ICNN surrogate (energy-based)

```python
import hyper_surrogate as hs

material = hs.NeoHooke({"C10": 0.5, "KBULK": 1000.0})
train_ds, val_ds, in_norm, out_norm = hs.create_datasets(
    material, n_samples=5000, input_type="invariants", target_type="energy",
)

# ICNN guarantees convexity of the predicted strain energy
model = hs.ICNN(input_dim=3, hidden_dims=[32, 32])
result = hs.Trainer(
    model, train_ds, val_ds,
    loss_fn=hs.EnergyStressLoss(alpha=1.0, beta=1.0),
    max_epochs=500,
).fit()

exported = hs.extract_weights(result.model, in_norm, out_norm)
exported.save("icnn_surrogate.npz")
```

See the [examples](https://jpsferreira.github.io/hyper-surrogate/examples/) for more usage patterns, including hybrid UMAT export and analytical Fortran generation.

## Installation

```bash
pip install hyper-surrogate        # core (NumPy, SymPy)
pip install hyper-surrogate[ml]    # with PyTorch for ML surrogates
```

From source with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/jpsferreira/hyper-surrogate.git
cd hyper-surrogate
uv sync --all-groups --extra ml
```

## Examples

Runnable scripts are in the [`examples/`](examples/) directory:

| Script                     | Description                                     |
| -------------------------- | ----------------------------------------------- |
| `train_neohooke_sef.py`    | Train MLP on NeoHooke SEF with hybrid inference |
| `train_neohooke_stress.py` | Train MLP on PK2 stress with `StressLoss`       |
| `train_icnn_energy.py`     | Train ICNN with `EnergyStressLoss`              |
| `export_hybrid_umat.py`    | End-to-end train + `HybridUMATEmitter` export   |
| `analytical_umat.py`       | Symbolic material to Fortran via `UMATHandler`  |

Run any example with:

```bash
uv run python examples/<script>.py
```

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for setup instructions and guidelines.
