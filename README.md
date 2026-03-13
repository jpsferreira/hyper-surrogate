# hyper-surrogate

[![Release](https://img.shields.io/github/v/release/jpsferreira/hyper-surrogate)](https://img.shields.io/github/v/release/jpsferreira/hyper-surrogate)
[![Build status](https://img.shields.io/github/actions/workflow/status/jpsferreira/hyper-surrogate/main.yml?branch=main)](https://github.com/jpsferreira/hyper-surrogate/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/jpsferreira/hyper-surrogate/branch/main/graph/badge.svg)](https://codecov.io/gh/jpsferreira/hyper-surrogate)
[![License](https://img.shields.io/github/license/jpsferreira/hyper-surrogate)](https://img.shields.io/github/license/jpsferreira/hyper-surrogate)

Data-driven surrogates for hyperelastic constitutive models in finite element analysis.

- **Github repository**: <https://github.com/jpsferreira/hyper-surrogate/>
- **Documentation**: <https://jpsferreira.github.io/hyper-surrogate/>

## Features

- **Symbolic mechanics** -- Automatic PK2 stress and stiffness tensor derivation via SymPy
- **ML surrogates** -- MLP and Input-Convex Neural Network (ICNN) models for constitutive response approximation
- **Fortran export** -- Transpile trained models to standalone Fortran 90 subroutines with baked-in weights
- **Analytical UMAT** -- Generate optimized Fortran UMAT subroutines from symbolic expressions

## Quick start

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

See the [examples](https://jpsferreira.github.io/hyper-surrogate/examples/) for more usage patterns.

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
