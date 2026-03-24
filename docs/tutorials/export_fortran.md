# Exporting to Fortran

This tutorial covers the three Fortran export pathways in **hyper-surrogate**: standalone NN modules, hybrid UMAT subroutines (NN energy + analytical mechanics), and purely analytical UMATs from symbolic material models.

---

## Overview: Three Export Paths

| Export Path         | Class               | What It Generates                                            | NN Required? |
| ------------------- | ------------------- | ------------------------------------------------------------ | :----------: |
| **Standalone NN**   | `FortranEmitter`    | Fortran 90 module with NN forward pass                       |     Yes      |
| **Hybrid UMAT**     | `HybridUMATEmitter` | Complete Abaqus UMAT (NN energy + analytical stress/tangent) |     Yes      |
| **Analytical UMAT** | `UMATHandler`       | Complete Abaqus UMAT from symbolic material (SymPy CSE)      |      No      |

```
              ┌─────────────────┐
              │  Trained Model  │
              └────────┬────────┘
                       │
           ┌───────────┼───────────┐
           ▼           ▼           ▼
    FortranEmitter  HybridUMAT  UMATHandler
     (standalone)   Emitter     (symbolic)
           │           │           │
           ▼           ▼           ▼
      nn_module.f90  umat.f90   umat.f
```

---

## 1. Weight Extraction: `ExportedModel`

Before any Fortran export, you need to extract the trained model's weights and normalizers into a portable format:

```python
import hyper_surrogate as hs

# After training...
exported = hs.extract_weights(result.model, in_norm, out_norm)

# Save to disk (NumPy .npz)
exported.save("my_model.npz")

# Load later
exported = hs.ExportedModel.load("my_model.npz")
```

### What's inside `ExportedModel`

| Field               | Type                 | Description                                       |
| ------------------- | -------------------- | ------------------------------------------------- |
| `layers`            | `List[LayerInfo]`    | Layer metadata (activation, dimensions)           |
| `weights`           | `Dict[str, ndarray]` | Weight matrices and bias vectors                  |
| `input_normalizer`  | `Dict`               | Input mean/std for standardization                |
| `output_normalizer` | `Dict`               | Output mean/std for de-standardization            |
| `metadata`          | `Dict`               | Architecture type, input/output dims, branch info |

---

## 2. Standalone Fortran Module (`FortranEmitter`)

Generates a self-contained Fortran 90 module with the NN forward pass. All weights are baked in as `PARAMETER` arrays.

### 2.1 MLP Export

```python
import hyper_surrogate as hs

# Train an MLP
material = hs.NeoHooke({"C10": 0.5, "KBULK": 1000.0})
train_ds, val_ds, in_norm, out_norm = hs.create_datasets(
    material, n_samples=5000,
    input_type="invariants", target_type="pk2_voigt",
)

model = hs.MLP(input_dim=3, output_dim=6, hidden_dims=[32, 32], activation="tanh")
result = hs.Trainer(model, train_ds, val_ds, loss_fn=hs.StressLoss(), max_epochs=500).fit()

# Export
exported = hs.extract_weights(result.model, in_norm, out_norm)
emitter = hs.FortranEmitter(exported)
emitter.write("nn_surrogate.f90")
```

### 2.2 What the generated Fortran looks like

The `.f90` file contains:

```fortran
MODULE nn_surrogate
  IMPLICIT NONE
  ! ── Normalization parameters ──
  DOUBLE PRECISION, PARAMETER :: INPUT_MEAN(3) = (/ ... /)
  DOUBLE PRECISION, PARAMETER :: INPUT_STD(3) = (/ ... /)
  DOUBLE PRECISION, PARAMETER :: OUTPUT_MEAN(6) = (/ ... /)
  DOUBLE PRECISION, PARAMETER :: OUTPUT_STD(6) = (/ ... /)

  ! ── Layer weights (15-digit precision) ──
  DOUBLE PRECISION, PARAMETER :: W1(32, 3) = RESHAPE((/ ... /), (/32, 3/))
  DOUBLE PRECISION, PARAMETER :: B1(32) = (/ ... /)
  ! ... more layers ...

CONTAINS
  SUBROUTINE nn_forward(input, output)
    DOUBLE PRECISION, INTENT(IN)  :: input(3)
    DOUBLE PRECISION, INTENT(OUT) :: output(6)
    DOUBLE PRECISION :: x_norm(3), h1(32), h2(32)

    ! Normalize input
    x_norm = (input - INPUT_MEAN) / INPUT_STD

    ! Layer 1: MATMUL + tanh
    h1 = TANH(MATMUL(W1, x_norm) + B1)

    ! Layer 2: MATMUL + tanh
    h2 = TANH(MATMUL(W2, h1) + B2)

    ! Output layer: MATMUL (no activation)
    output = MATMUL(W3, h2) + B3

    ! Denormalize output
    output = output * OUTPUT_STD + OUTPUT_MEAN
  END SUBROUTINE
END MODULE
```

### 2.3 ICNN and PolyconvexICNN Export

The `FortranEmitter` auto-detects the architecture and generates the appropriate forward pass:

```python
# ICNN export (includes softplus enforcement on z-path weights)
model = hs.ICNN(input_dim=3, hidden_dims=[32, 32])
# ... train ...
exported = hs.extract_weights(result.model, in_norm, out_norm)
hs.FortranEmitter(exported).write("icnn_energy.f90")

# PolyconvexICNN export (per-branch forward pass)
model = hs.PolyconvexICNN(groups=[[0], [1], [2]], hidden_dims=[32, 32])
# ... train ...
exported = hs.extract_weights(result.model, in_norm, out_norm)
hs.FortranEmitter(exported).write("polyconvex_energy.f90")
```

### Key properties of generated Fortran

| Property         | Details                                            |
| ---------------- | -------------------------------------------------- |
| **Precision**    | 15-digit double precision (`:.15e` format)         |
| **Array layout** | Column-major (native Fortran ordering)             |
| **Forward pass** | `MATMUL`-based (efficient on modern compilers)     |
| **Dependencies** | None — fully self-contained                        |
| **Activations**  | `TANH`, `softplus` ($\ln(1 + e^x)$), ReLU, sigmoid |

---

## 3. Hybrid UMAT Emitter (`HybridUMATEmitter`)

The most powerful export: generates a **complete Abaqus UMAT subroutine** where:

- The **neural network** provides $W(\bar{I}_1, \bar{I}_2, J)$ (strain energy)
- **Analytical Fortran code** handles kinematics, stress, tangent, and push-forward

### 3.1 How it works

```
Abaqus calls UMAT with F (deformation gradient)
  │
  ├─ 1. Kinematics:  F → C = F^T·F → I1_bar, I2_bar, J (analytical)
  │
  ├─ 2. NN Forward:  [I1_bar, I2_bar, J] → normalize → hidden layers → W (energy)
  │
  ├─ 3. NN Backward: Backprop dW/dI1_bar, dW/dI2_bar, dW/dJ
  │
  ├─ 4. PK2 Stress:  S = 2·Σ(dW/dI_k)·(dI_k/dC)  (analytical chain rule)
  │
  ├─ 5. Hessian:     Forward-mode Jacobian for d²W/dI² (NN second derivatives)
  │
  ├─ 6. Tangent:     C_mat from dI/dC and d²W/dI²  (analytical)
  │
  ├─ 7. Push-forward: σ = (1/J)·F·S·F^T  (Cauchy stress)
  │
  └─ 8. Jaumann:     Spatial tangent + Jaumann rate correction
```

### 3.2 Example: Isotropic Hybrid UMAT

```python
import numpy as np
import hyper_surrogate as hs
from hyper_surrogate.data.dataset import MaterialDataset, Normalizer
from hyper_surrogate.data.deformation import DeformationGenerator
from hyper_surrogate.mechanics.kinematics import Kinematics

# 1. Generate data
material = hs.NeoHooke({"C10": 0.5, "KBULK": 1000.0})
n = 20000
gen = DeformationGenerator(seed=42)
F = gen.combined(n, stretch_range=(0.8, 1.3), shear_range=(-0.2, 0.2))
rng = np.random.default_rng(99)
J_target = rng.uniform(0.95, 1.05, size=n)
F = F * (J_target / np.linalg.det(F))[:, None, None] ** (1.0 / 3.0)
C = Kinematics.right_cauchy_green(F)

# 2. Invariants and targets
i1 = Kinematics.isochoric_invariant1(C)
i2 = Kinematics.isochoric_invariant2(C)
j = np.sqrt(Kinematics.det_invariant(C))
inputs = np.column_stack([i1, i2, j])
energy = material.evaluate_energy(C)
dW_dI = material.evaluate_energy_grad_invariants(C)

# 3. Normalize and build datasets
in_norm = Normalizer().fit(inputs)
X = in_norm.transform(inputs).astype(np.float32)
W = energy.reshape(-1, 1).astype(np.float32)
S = (dW_dI * in_norm.params["std"]).astype(np.float32)

n_val = int(n * 0.15)
idx = np.random.default_rng(42).permutation(n)
train_ds = MaterialDataset(X[idx[n_val:]], (W[idx[n_val:]], S[idx[n_val:]]))
val_ds = MaterialDataset(X[idx[:n_val]], (W[idx[:n_val]], S[idx[:n_val]]))

# 4. Train
model = hs.MLP(input_dim=3, output_dim=1, hidden_dims=[64, 64, 64], activation="softplus")
result = hs.Trainer(
    model, train_ds, val_ds,
    loss_fn=hs.EnergyStressLoss(alpha=1.0, beta=1.0),
    max_epochs=3000, lr=1e-3, patience=300, batch_size=512,
).fit()

# 5. Export hybrid UMAT
energy_norm = Normalizer().fit(energy.reshape(-1, 1))
exported = hs.extract_weights(result.model, in_norm, energy_norm)
emitter = hs.HybridUMATEmitter(exported)
emitter.write("hybrid_umat.f90")
```

### 3.3 Anisotropic Hybrid UMAT (5D invariants)

When `input_dim=5`, the emitter auto-detects anisotropic mode and generates fiber invariant computation ($I_4, I_5$) plus the corresponding derivatives ($\partial I_4 / \partial \mathbf{C}$, $\partial I_5 / \partial \mathbf{C}$):

```python
# After training with 5D invariants (I1_bar, I2_bar, J, I4, I5)...
exported = hs.extract_weights(result.model, in_norm, energy_norm)
emitter = hs.HybridUMATEmitter(exported)
emitter.write("hybrid_umat_aniso.f90")
# Fiber direction is passed via PROPS(1:3) in the Abaqus input file
```

### 3.4 PolyconvexICNN Hybrid UMAT

```python
# After training a PolyconvexICNN...
exported = hs.extract_weights(result.model, in_norm, energy_norm)
emitter = hs.HybridUMATEmitter(exported)
emitter.write("hybrid_umat_polyconvex.f90")
# Block-diagonal Hessian: more efficient tangent computation
```

### 3.5 What the hybrid UMAT contains

| Section                  | Description                                                                                            |
| ------------------------ | ------------------------------------------------------------------------------------------------------ |
| **Properties block**     | Material parameters from `PROPS` array                                                                 |
| **Kinematics**           | $\mathbf{F} \to \mathbf{C} \to \bar{I}_1, \bar{I}_2, J$ (+ $I_4, I_5$ for anisotropic)                 |
| **NN Forward pass**      | Normalized input → hidden layers → $W$ (energy scalar)                                                 |
| **NN Backward pass**     | Backpropagation: $\partial W / \partial I_\alpha$                                                      |
| **Jacobian propagation** | Forward-mode AD: $\partial^2 W / \partial I_\alpha \partial I_\beta$ (Hessian)                         |
| **PK2 stress**           | $\mathbf{S} = 2 \sum_\alpha (\partial W / \partial I_\alpha)(\partial I_\alpha / \partial \mathbf{C})$ |
| **Cauchy push-forward**  | $\boldsymbol{\sigma} = (1/J)\,\mathbf{F}\,\mathbf{S}\,\mathbf{F}^T$                                    |
| **Spatial tangent**      | Full 4th-order tensor push-forward + Jaumann rate correction                                           |

---

## 4. Analytical UMAT (`UMATHandler`)

For when you want a Fortran UMAT directly from a symbolic material definition — **no neural network involved**:

```python
from hyper_surrogate.mechanics.materials import NeoHooke
from hyper_surrogate.export.fortran.analytical import UMATHandler

material = NeoHooke({"C10": 0.5, "KBULK": 1000.0})
handler = UMATHandler(material)
handler.generate("neohooke_umat.f")

print("Generated: neohooke_umat.f")
```

### How it works

```
Material (symbolic SEF)
  │
  ├─ SymPy: W(C) → S = ∂W/∂C (symbolic differentiation)
  │
  ├─ SymPy: S(C) → C_mat = ∂S/∂C (symbolic 2nd derivative)
  │
  ├─ SymPy CSE: Common Subexpression Elimination
  │
  └─ Fortran code generation: optimized UMAT subroutine
```

### Features

| Feature               | Details                                                 |
| --------------------- | ------------------------------------------------------- |
| **Exact derivatives** | Symbolic differentiation (no numerical errors)          |
| **CSE optimization**  | SymPy identifies and eliminates repeated subexpressions |
| **Cauchy stress**     | Push-forward from PK2 to Cauchy                         |
| **Spatial tangent**   | Full tangent with Jaumann rate correction               |
| **Compatible**        | Works with any symbolic `Material` in the library       |

### Works with any material

```python
from hyper_surrogate.mechanics.materials import MooneyRivlin, Yeoh, Demiray

# Mooney-Rivlin
mat = MooneyRivlin({"C10": 0.3, "C01": 0.2, "KBULK": 1000.0})
UMATHandler(mat).generate("mooneyrivlin_umat.f")

# Yeoh
mat = Yeoh({"C10": 0.5, "C20": -0.01, "C30": 0.001, "KBULK": 1000.0})
UMATHandler(mat).generate("yeoh_umat.f")

# Demiray
mat = Demiray({"C1": 0.05, "C2": 8.0, "KBULK": 1000.0})
UMATHandler(mat).generate("demiray_umat.f")
```

---

## 5. Choosing an Export Path

| Question                                  | Answer → Export Path                                   |
| ----------------------------------------- | ------------------------------------------------------ |
| Do you need a NN surrogate?               | **No** → `UMATHandler` (analytical)                    |
| Do you need stress + tangent for FE?      | **Yes** → `HybridUMATEmitter`                          |
| Do you only need the NN forward pass?     | **Yes** → `FortranEmitter` (standalone)                |
| Is the material energy-based (scalar NN)? | **Yes** → `HybridUMATEmitter`                          |
| Is the material stress-based (6D NN)?     | → `FortranEmitter` + custom UMAT wrapper               |
| Is thermodynamic consistency critical?    | **Yes** → `HybridUMATEmitter` with ICNN/PolyconvexICNN |

### Decision flowchart

```
Need a surrogate?
├── No  ──────────────> UMATHandler (analytical, symbolic)
└── Yes
    └── Need full UMAT (stress + tangent)?
        ├── No  ──────> FortranEmitter (standalone NN module)
        └── Yes ──────> HybridUMATEmitter (NN energy + analytical mechanics)
```

---

## 6. Using the Generated UMAT in Abaqus

### Abaqus input file setup

```
*MATERIAL, NAME=NN_SURROGATE
*DEPVAR
  1,
*USER MATERIAL, CONSTANTS=2
** KBULK, (unused)
  1000.0, 0.0
```

For anisotropic models, pass the fiber direction via `PROPS`:

```
*USER MATERIAL, CONSTANTS=5
** KBULK, fiber_x, fiber_y, fiber_z, (unused)
  1000.0, 1.0, 0.0, 0.0, 0.0
```

### Compilation

```bash
# Compile the UMAT with Abaqus
abaqus job=my_simulation user=hybrid_umat.f90

# Or with Intel Fortran directly
ifort -c -O2 hybrid_umat.f90
```

The generated code uses only standard Fortran 90 features and requires no external libraries.
