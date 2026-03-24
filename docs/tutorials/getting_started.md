# Getting Started

This tutorial walks you through installing **hyper-surrogate**, creating your first material, computing stress, and understanding the core workflow.

---

## Installation

### Prerequisites

- **Python 3.12+**
- **uv** (recommended) or pip

### Install with uv (recommended)

```bash
# Clone the repository
git clone https://github.com/jpsferreira/hyper-surrogate.git
cd hyper-surrogate

# Install core dependencies (NumPy, SymPy only — no PyTorch)
uv sync

# Install with ML support (adds PyTorch)
uv sync --all-groups --extra ml

# Install pre-commit hooks and dev tools
make install
```

### Install with pip

```bash
pip install hyper-surrogate          # Core only
pip install hyper-surrogate[ml]      # Core + PyTorch
```

### Verify installation

```python
import hyper_surrogate as hs
print(hs.__version__)
```

### Dependency overview

| Component              | Required Packages                 | Install Command        |
| ---------------------- | --------------------------------- | ---------------------- |
| Core (mechanics, data) | `numpy`, `sympy`                  | `uv sync`              |
| ML (models, training)  | `torch`                           | `uv sync --extra ml`   |
| Visualization          | `matplotlib`                      | Included in core       |
| Parameter fitting      | `scipy`                           | Included in core       |
| Documentation          | `mkdocs-material`, `mkdocstrings` | `uv sync --all-groups` |

---

## Your First Material

Every workflow starts by defining a **Material**. The simplest is `NeoHooke`:

```python
import numpy as np
import hyper_surrogate as hs

# Create a Neo-Hookean material
# C10 = half the shear modulus (μ/2)
# KBULK = bulk modulus (enforces near-incompressibility)
material = hs.NeoHooke(parameters={"C10": 0.5, "KBULK": 1000.0})
```

The material object stores:

- The **symbolic strain energy function** $W(\mathbf{C})$
- Auto-derived **PK2 stress** $\mathbf{S} = 2\,\partial W / \partial \mathbf{C}$
- Auto-derived **material tangent** $\mathbb{C} = 4\,\partial^2 W / \partial \mathbf{C}^2$

All derivatives are computed symbolically via SymPy — no numerical approximations.

---

## Computing Stress and Energy

### Step 1: Create a deformation

```python
# A simple uniaxial stretch: λ=1.2 along axis 1
lam = 1.2
F = np.array([[[lam, 0, 0],
               [0, 1.0/np.sqrt(lam), 0],
               [0, 0, 1.0/np.sqrt(lam)]]])  # shape (1, 3, 3)
```

### Step 2: Compute the right Cauchy-Green tensor

```python
C = hs.Kinematics.right_cauchy_green(F)  # C = F^T F, shape (1, 3, 3)
```

### Step 3: Evaluate stress and energy

```python
# Second Piola-Kirchhoff stress
pk2 = material.evaluate_pk2(C)       # shape (1, 3, 3)

# Strain energy density
energy = material.evaluate_energy(C)  # shape (1,)

# Material tangent (elasticity tensor)
cmat = material.evaluate_cmat(C)      # shape (1, 3, 3, 3, 3)

print(f"Energy:     {energy[0]:.6f}")
print(f"PK2 (S11):  {pk2[0, 0, 0]:.6f}")
print(f"PK2 (S22):  {pk2[0, 1, 1]:.6f}")
```

### Step 4: Batch evaluation

All operations are vectorized — pass `N` samples at once:

```python
# Generate 1000 random uniaxial deformations
gen = hs.DeformationGenerator(seed=42)
F_batch = gen.uniaxial(n=1000)          # (1000, 3, 3)
C_batch = hs.Kinematics.right_cauchy_green(F_batch)

pk2_batch = material.evaluate_pk2(C_batch)        # (1000, 3, 3)
energy_batch = material.evaluate_energy(C_batch)  # (1000,)

print(f"Energy range: [{energy_batch.min():.4f}, {energy_batch.max():.4f}]")
```

---

## Kinematics Utilities

The `Kinematics` class provides batch-vectorized continuum mechanics operations:

```python
F = gen.combined(n=500)  # Mixed deformation modes

# Strain tensors
C = hs.Kinematics.right_cauchy_green(F)  # (500, 3, 3)
B = hs.Kinematics.left_cauchy_green(F)   # (500, 3, 3)

# Volume ratio
J = hs.Kinematics.jacobian(F)  # (500,)

# Isochoric invariants
I1_bar = hs.Kinematics.isochoric_invariant1(C)  # (500,)
I2_bar = hs.Kinematics.isochoric_invariant2(C)  # (500,)

# Principal stretches (sorted descending)
stretches = hs.Kinematics.principal_stretches(C)  # (500, 3)

# Fiber invariants (for anisotropic materials)
a0 = np.array([1.0, 0.0, 0.0])
I4 = hs.Kinematics.fiber_invariant4(C, a0)  # (500,)
I5 = hs.Kinematics.fiber_invariant5(C, a0)  # (500,)
```

**Quick reference table:**

| Method                    | Input     | Output Shape | Description                                         |
| ------------------------- | --------- | ------------ | --------------------------------------------------- |
| `right_cauchy_green(F)`   | `(N,3,3)` | `(N,3,3)`    | $\mathbf{C} = \mathbf{F}^T\mathbf{F}$               |
| `left_cauchy_green(F)`    | `(N,3,3)` | `(N,3,3)`    | $\mathbf{b} = \mathbf{F}\mathbf{F}^T$               |
| `jacobian(F)`             | `(N,3,3)` | `(N,)`       | $J = \det(\mathbf{F})$                              |
| `isochoric_invariant1(C)` | `(N,3,3)` | `(N,)`       | $\bar{I}_1 = J^{-2/3} \text{tr}(\mathbf{C})$        |
| `isochoric_invariant2(C)` | `(N,3,3)` | `(N,)`       | $\bar{I}_2 = J^{-4/3} I_2$                          |
| `det_invariant(C)`        | `(N,3,3)` | `(N,)`       | $I_3 = \det(\mathbf{C}) = J^2$                      |
| `principal_stretches(C)`  | `(N,3,3)` | `(N,3)`      | $\lambda_1 \ge \lambda_2 \ge \lambda_3$             |
| `fiber_invariant4(C, a0)` | `(N,3,3)` | `(N,)`       | $I_4 = \mathbf{a}_0 \cdot \mathbf{C}\mathbf{a}_0$   |
| `fiber_invariant5(C, a0)` | `(N,3,3)` | `(N,)`       | $I_5 = \mathbf{a}_0 \cdot \mathbf{C}^2\mathbf{a}_0$ |

---

## Available Material Models

Here is every model you can instantiate:

| Model                  | Import                              | Invariants                 | Parameters                   |
| ---------------------- | ----------------------------------- | -------------------------- | ---------------------------- |
| `NeoHooke`             | `hs.NeoHooke`                       | $\bar{I}_1$                | `C10, KBULK`                 |
| `MooneyRivlin`         | `hs.MooneyRivlin`                   | $\bar{I}_1, \bar{I}_2$     | `C10, C01, KBULK`            |
| `Yeoh`                 | `from ...materials import Yeoh`     | $\bar{I}_1$                | `C10, C20, C30, KBULK`       |
| `Demiray`              | `from ...materials import Demiray`  | $\bar{I}_1$                | `C1, C2, KBULK`              |
| `Ogden`                | `from ...materials import Ogden`    | $\bar{\lambda}_i$          | `mu_p, alpha_p, KBULK`       |
| `Fung`                 | `from ...materials import Fung`     | $\mathbf{E}$               | `c, b1, b2, KBULK`           |
| `HolzapfelOgden`       | `hs.HolzapfelOgden`                 | $\bar{I}_1, I_4$           | `a, b, af, bf, KBULK`        |
| `GasserOgdenHolzapfel` | `from ...materials import GOH`      | $\bar{I}_1, I_4$           | `a, b, af, bf, kappa, KBULK` |
| `Guccione`             | `from ...materials import Guccione` | $\mathbf{E}$ (fiber frame) | `C, bf, bt, bfs, KBULK`      |

---

## The Big Picture: End-to-End Pipeline

```
┌──────────┐    ┌────────────────────┐    ┌──────────┐    ┌─────────┐    ┌──────────────┐
│ Material │───>│ DeformationGenerator│───>│ Dataset  │───>│ Trainer │───>│ FortranEmitter│
│ (SEF)    │    │ (F, C, invariants) │    │ (X, Y)   │    │ (model) │    │ (.f90 UMAT)  │
└──────────┘    └────────────────────┘    └──────────┘    └─────────┘    └──────────────┘
```

Each stage is covered in detail in the following tutorials:

| Tutorial                                          | What You'll Learn                               |
| ------------------------------------------------- | ----------------------------------------------- |
| [Data Generation](data_generation.md)             | Deformation modes, invariants, datasets         |
| [Training Surrogates](training_surrogates.md)     | MLP, ICNN, PolyconvexICNN, loss functions       |
| [Fortran Export](export_fortran.md)               | Standalone NN, hybrid UMAT, analytical UMAT     |
| [Anisotropic Materials](anisotropic_materials.md) | Fibers, arterial walls, cardiac tissue          |
| [Advanced Topics](advanced_topics.md)             | CANN discovery, parameter fitting, benchmarking |

---

## Minimal End-to-End Example

From material definition to Fortran in ~15 lines:

```python
import hyper_surrogate as hs

# 1. Material
material = hs.NeoHooke({"C10": 0.5, "KBULK": 1000.0})

# 2. Dataset (invariants -> PK2 stress in Voigt notation)
train_ds, val_ds, in_norm, out_norm = hs.create_datasets(
    material, n_samples=5000,
    input_type="invariants", target_type="pk2_voigt",
)

# 3. Train
model = hs.MLP(input_dim=3, output_dim=6, hidden_dims=[32, 32], activation="tanh")
result = hs.Trainer(model, train_ds, val_ds, loss_fn=hs.StressLoss(), max_epochs=500).fit()
print(f"Best val loss: {result.history['val_loss'][result.best_epoch]:.6f}")

# 4. Export to Fortran
exported = hs.extract_weights(result.model, in_norm, out_norm)
emitter = hs.FortranEmitter(exported)
emitter.write("nn_surrogate.f90")
```

The generated `nn_surrogate.f90` is a self-contained Fortran 90 module with all weights baked in as `PARAMETER` arrays — no external dependencies, ready for any FE solver.
