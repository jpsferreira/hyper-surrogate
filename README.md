# hyper-surrogate

[![Release](https://img.shields.io/github/v/release/jpsferreira/hyper-surrogate)](https://img.shields.io/github/v/release/jpsferreira/hyper-surrogate)
[![Build status](https://img.shields.io/github/actions/workflow/status/jpsferreira/hyper-surrogate/main.yml?branch=main)](https://github.com/jpsferreira/hyper-surrogate/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/jpsferreira/hyper-surrogate/branch/main/graph/badge.svg)](https://codecov.io/gh/jpsferreira/hyper-surrogate)
[![License](https://img.shields.io/github/license/jpsferreira/hyper-surrogate)](https://img.shields.io/github/license/jpsferreira/hyper-surrogate)

**Define hyperelastic materials in Python, deploy them in your finite element solver.**

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

Finite element solvers need constitutive models. Writing and maintaining
those models — especially user-material subroutines for Abaqus, FEAP and
similar Fortran-based solvers — is repetitive: kinematics, second
Piola–Kirchhoff stress, consistent tangent, Jaumann correction,
reference-configuration freedom, all in Fortran 77 or 90.

**hyper-surrogate** lets you define your material once in Python and
emit a self-contained Fortran 90 user subroutine deployable in any
Fortran-based FE solver. You choose how the material is defined:

1. **Built-in symbolic SEF** — one of ten classical isotropic and
   anisotropic models (Neo-Hooke, Mooney-Rivlin, Yeoh, Holzapfel-Ogden,
   Gasser-Ogden-Holzapfel, …).
2. **Custom symbolic SEF** — subclass `Material` with a `sef` property
   in `SymPy` and the framework derives stress and tangent automatically.
3. **Data-driven surrogate** — train an MLP, ICNN, or polyconvex ICNN
   on energy and stress samples from any ground-truth source and emit
   a hybrid UMAT whose energy is the trained network and whose stress
   and tangent are computed analytically in Fortran.

All three paths produce the same artefact: a `.f90` with Cauchy stress,
analytical consistent tangent, and stress-freedom at the reference
configuration enforced exactly at emission time. No Python runtime
is required at solve time.

## Architecture

```
Material ─> DeformationGenerator ─> Dataset ─> MLP / ICNN ─> HybridUMATEmitter ─> .f90
   │                                              │
   │         (symbolic SEF)                        │  (trained NN weights)
   │                                              │
   └──── UMATHandler ─────────────────────────────┘
         (analytical Fortran via SymPy CSE)
```

## Features

- **Three entry points, one output** — built-in SEF, custom SymPy SEF,
  or trained surrogate, all emit a structurally identical Fortran UMAT.
- **Symbolic mechanics** — automatic PK2 stress and stiffness tensor
  derivation via SymPy; analytical Cauchy push-forward and Jaumann
  correction emitted with common-subexpression elimination.
- **NN architectures** — MLP, Input-Convex NN, and polyconvex ICNN,
  trained with an energy-plus-stress dual loss that backpropagates
  the gradient target through autograd.
- **Hybrid UMAT** — neural-network strain energy with closed-form
  analytical kinematics, stress, and consistent tangent in Fortran;
  layer-additive Hessian backpropagation verified against autograd
  at $10^{-5}$ tolerance.
- **Stress freedom at reference** — enforced exactly at emission time
  by subtracting a compile-time offset; no surprise residual stress
  at $\mathbf{C} = \mathbf{I}$.
- **Anisotropic support** — fiber pseudo-invariants $I_4, I_5$ for one
  or two fiber families with arbitrary orientations.

## Quick Start

Three short pipelines, mirroring the three ways to define a material:

### Path A — built-in symbolic SEF

```python
from hyper_surrogate import NeoHooke, UMATHandler

UMATHandler(NeoHooke({"C10": 0.5, "KBULK": 1000.0})).generate("neohooke.f90")
```

### Path B — custom user-defined SEF

```python
import sympy as sp
from hyper_surrogate import Material, UMATHandler

class MyOgdenLike(Material):
    DEFAULT_PARAMS = {"mu": 1.0, "alpha": 2.0, "KBULK": 1000.0}

    def __init__(self, parameters=None):
        super().__init__({**self.DEFAULT_PARAMS, **(parameters or {})})

    @property
    def sef(self):
        h = self._handler
        mu, alpha, K = (self._symbols[k] for k in ("mu", "alpha", "KBULK"))
        return ((mu / alpha) * (h.isochoric_invariant1 ** (alpha / 2) - 3)
                + 0.5 * K * (sp.sqrt(h.invariant3) - 1) ** 2)

UMATHandler(MyOgdenLike()).generate("mysef.f90")
```

### Path C — data-driven surrogate

```python
from hyper_surrogate import (
    NeoHooke, MLP, Trainer, EnergyStressLoss,
    create_datasets, extract_weights, HybridUMATEmitter,
)

material = NeoHooke({"C10": 0.5, "KBULK": 1000.0})
train_ds, val_ds, in_norm, energy_norm = create_datasets(
    material, n_samples=4000, input_type="invariants",
    target_type="energy", deformation_mode="combined_compressible",
)
model = MLP(input_dim=3, output_dim=1, hidden_dims=[64, 64, 64], activation="softplus")
result = Trainer(model, train_ds, val_ds,
                 loss_fn=EnergyStressLoss(alpha=1.0, beta=1.0),
                 max_epochs=2000, patience=200).fit()
exported = extract_weights(result.model, in_norm, energy_norm)
HybridUMATEmitter(exported).write("neohooke_hybrid.f90")
```

See the [documentation](https://jpsferreira.github.io/hyper-surrogate/)
for the full tutorial set, including [custom materials](https://jpsferreira.github.io/hyper-surrogate/tutorials/custom_materials/),
[anisotropic models](https://jpsferreira.github.io/hyper-surrogate/tutorials/anisotropic_materials/),
and the [export-path decision tree](https://jpsferreira.github.io/hyper-surrogate/tutorials/export_fortran/).

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

| Script                     | Path | Description                                             |
| -------------------------- | ---- | ------------------------------------------------------- |
| `analytical_umat.py`       | A    | Built-in `NeoHooke` → analytical UMAT via `UMATHandler` |
| `custom_sef.py`            | B    | Custom Ogden-like SEF → analytical UMAT                 |
| `export_hybrid_umat.py`    | C    | End-to-end train + `HybridUMATEmitter` export           |
| `train_neohooke_sef.py`    | C    | Train MLP on NeoHooke SEF with hybrid inference         |
| `train_icnn_energy.py`     | C    | Train ICNN with `EnergyStressLoss`                      |
| `train_polyconvex.py`      | C    | Train PolyconvexICNN with per-invariant convexity       |
| `train_holzapfel_ogden.py` | C    | Anisotropic fiber-reinforced training pipeline          |

Run any example with:

```bash
uv run python examples/<script>.py
```

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for
setup instructions and guidelines.
