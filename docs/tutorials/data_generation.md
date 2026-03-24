# Data Generation

This tutorial covers the full data generation pipeline: creating synthetic deformation gradients, computing invariants, building normalized datasets, and preparing training data for neural network surrogates.

---

## Overview

The data pipeline consists of three stages:

```
DeformationGenerator  ──>  Kinematics (invariants)  ──>  create_datasets()
      (F tensors)           (I1_bar, I2_bar, J, ...)      (train_ds, val_ds)
```

---

## 1. Deformation Generator

`DeformationGenerator` creates batches of deformation gradient tensors $\mathbf{F}$ representing different loading modes.

```python
import numpy as np
from hyper_surrogate.data.deformation import DeformationGenerator

gen = DeformationGenerator(seed=42)  # Reproducible random generation
```

### 1.1 Uniaxial Tension/Compression

Stretch $\lambda$ along axis 1, with transverse contraction to preserve volume:

$$\mathbf{F} = \begin{bmatrix} \lambda & 0 & 0 \\ 0 & \lambda^{-1/2} & 0 \\ 0 & 0 & \lambda^{-1/2} \end{bmatrix}$$

```python
F_uni = gen.uniaxial(n=1000, stretch_range=(0.7, 1.5))
print(f"Shape: {F_uni.shape}")  # (1000, 3, 3)

# Verify incompressibility
J = np.linalg.det(F_uni)
print(f"J range: [{J.min():.6f}, {J.max():.6f}]")  # ≈ [1.0, 1.0]
```

### 1.2 Biaxial Stretch

Independent stretches $\lambda_1, \lambda_2$ along axes 1 and 2:

$$\mathbf{F} = \begin{bmatrix} \lambda_1 & 0 & 0 \\ 0 & \lambda_2 & 0 \\ 0 & 0 & (\lambda_1 \lambda_2)^{-1} \end{bmatrix}$$

```python
F_bi = gen.biaxial(n=1000, stretch_range=(0.8, 1.3))
```

### 1.3 Simple Shear

Shear deformation in the 1-2 plane:

$$\mathbf{F} = \begin{bmatrix} 1 & \gamma & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}$$

```python
F_shear = gen.shear(n=1000, shear_range=(-0.3, 0.3))
```

### 1.4 Combined Deformations

The most realistic mode — combines biaxial stretch, uniaxial stretch, and shear with random rotations:

```python
F_combined = gen.combined(
    n=10000,
    stretch_range=(0.8, 1.3),
    shear_range=(-0.2, 0.2),
)
```

This is the **recommended mode for training data**: it covers a wide range of deformation states and ensures the surrogate generalizes well.

### Summary of Deformation Modes

| Mode | Function | Key Parameters | Volume-Preserving |
|------|----------|---------------|:-----------------:|
| Uniaxial | `gen.uniaxial()` | `stretch_range` | Yes |
| Biaxial | `gen.biaxial()` | `stretch_range` | Yes |
| Shear | `gen.shear()` | `shear_range` | Yes |
| Combined | `gen.combined()` | `stretch_range`, `shear_range` | Yes |

---

## 2. Adding Volumetric Perturbation

All basic deformation modes are **incompressible** ($J = 1$). For nearly-incompressible materials (which is what we simulate in FE), you need to add volumetric perturbation so the model learns the volumetric response:

```python
n = 10000
F = gen.combined(n, stretch_range=(0.8, 1.3), shear_range=(-0.2, 0.2))

# Perturb volume ratio to J ∈ [0.95, 1.05]
rng = np.random.default_rng(99)
J_target = rng.uniform(0.95, 1.05, size=n)
J_current = np.linalg.det(F)
F = F * (J_target / J_current)[:, None, None] ** (1.0 / 3.0)

# Verify
J_new = np.linalg.det(F)
print(f"J range: [{J_new.min():.4f}, {J_new.max():.4f}]")
# Output: J range: [0.9500, 1.0500]
```

The scaling factor $(J_{\text{target}} / J_{\text{current}})^{1/3}$ applies a uniform dilation to each deformation gradient.

---

## 3. Computing Invariants

Once you have deformation gradients, compute the invariants that serve as neural network inputs:

```python
from hyper_surrogate.mechanics.kinematics import Kinematics

C = Kinematics.right_cauchy_green(F)  # (N, 3, 3)

# Isochoric invariants (3 inputs for isotropic models)
i1_bar = Kinematics.isochoric_invariant1(C)  # (N,)
i2_bar = Kinematics.isochoric_invariant2(C)  # (N,)
j = np.sqrt(Kinematics.det_invariant(C))     # (N,)

inputs_iso = np.column_stack([i1_bar, i2_bar, j])  # (N, 3)
```

For **anisotropic** materials, add fiber invariants:

```python
fiber_dir = np.array([1.0, 0.0, 0.0])
i4 = Kinematics.fiber_invariant4(C, fiber_dir)  # (N,)
i5 = Kinematics.fiber_invariant5(C, fiber_dir)  # (N,)

inputs_aniso = np.column_stack([i1_bar, i2_bar, j, i4, i5])  # (N, 5)
```

### Invariant ranges (typical)

| Invariant | Reference Value ($\mathbf{F} = \mathbf{I}$) | Typical Training Range |
|-----------|:-------------------------------------------:|:---------------------:|
| $\bar{I}_1$ | 3.0 | 2.5 -- 4.0 |
| $\bar{I}_2$ | 3.0 | 2.5 -- 4.5 |
| $J$ | 1.0 | 0.9 -- 1.1 |
| $I_4$ | 1.0 | 0.5 -- 2.0 |
| $I_5$ | 1.0 | 0.3 -- 4.0 |

---

## 4. Computing Targets

The material object computes the training targets:

```python
import hyper_surrogate as hs

material = hs.NeoHooke({"C10": 0.5, "KBULK": 1000.0})

# Energy (scalar per sample)
energy = material.evaluate_energy(C)  # (N,)

# Energy gradient w.r.t. invariants (for thermodynamically consistent training)
dW_dI = material.evaluate_energy_grad_invariants(C)  # (N, 3) or (N, 5)

# PK2 stress (full tensor)
pk2 = material.evaluate_pk2(C)  # (N, 3, 3)

# Material tangent
cmat = material.evaluate_cmat(C)  # (N, 3, 3, 3, 3)
```

### Target types summary

| Target Type | Shape | Use Case |
|------------|-------|----------|
| `energy` | `(N,)` | Energy-based training (ICNN, hybrid UMAT) |
| `dW_dI` | `(N, n_inv)` | Stress gradient (for `EnergyStressLoss`) |
| PK2 stress | `(N, 3, 3)` | Direct stress prediction |
| Material tangent | `(N, 3, 3, 3, 3)` | Stress + tangent prediction |

---

## 5. Using `create_datasets()` (Recommended)

The `create_datasets()` factory function handles the entire pipeline — deformation generation, invariant computation, target evaluation, normalization, and train/val split — in a single call:

```python
import hyper_surrogate as hs

material = hs.NeoHooke({"C10": 0.5, "KBULK": 1000.0})

train_ds, val_ds, in_norm, out_norm = hs.create_datasets(
    material,
    n_samples=10000,
    input_type="invariants",    # "invariants" or "cauchy_green"
    target_type="pk2_voigt",    # "energy", "pk2_voigt", or "pk2_voigt+cmat_voigt"
    seed=42,
)

print(f"Training samples: {len(train_ds)}")
print(f"Validation samples: {len(val_ds)}")
print(f"Input shape: {train_ds.inputs.shape}")    # (N_train, 3)
print(f"Target shape: {train_ds.targets.shape}")   # (N_train, 6) for pk2_voigt
```

### Parameter reference

| Parameter | Options | Default | Description |
|-----------|---------|---------|-------------|
| `n_samples` | int | — | Total number of deformation samples |
| `input_type` | `"invariants"`, `"cauchy_green"` | `"invariants"` | NN input representation |
| `target_type` | `"energy"`, `"pk2_voigt"`, `"pk2_voigt+cmat_voigt"` | `"pk2_voigt"` | What the NN predicts |
| `seed` | int | `None` | Random seed for reproducibility |

### Input types

| `input_type` | Dimensions | Components |
|-------------|:----------:|-----------|
| `"invariants"` (isotropic) | 3 | $\bar{I}_1, \bar{I}_2, J$ |
| `"invariants"` (anisotropic) | 5 | $\bar{I}_1, \bar{I}_2, J, I_4, I_5$ |
| `"cauchy_green"` | 6 | $C_{11}, C_{22}, C_{33}, C_{12}, C_{13}, C_{23}$ (Voigt) |

### Target types

| `target_type` | Dimensions | Components |
|--------------|:----------:|-----------|
| `"energy"` | 1 | $W$ (+ gradient $\partial W/\partial I$ as auxiliary) |
| `"pk2_voigt"` | 6 | $S_{11}, S_{22}, S_{33}, S_{12}, S_{13}, S_{23}$ |
| `"pk2_voigt+cmat_voigt"` | 27 | 6 stress + 21 unique tangent components |

---

## 6. Normalization

All inputs and outputs are standardized (zero-mean, unit-variance) before training. The `Normalizer` stores the transformation parameters for export:

```python
from hyper_surrogate.data.dataset import Normalizer

# Automatic normalization inside create_datasets()
train_ds, val_ds, in_norm, out_norm = hs.create_datasets(material, n_samples=5000)

# Access normalization parameters
print(f"Input mean:  {in_norm.params['mean']}")
print(f"Input std:   {in_norm.params['std']}")
print(f"Output mean: {out_norm.params['mean']}")
print(f"Output std:  {out_norm.params['std']}")
```

### Manual normalization (for custom pipelines)

```python
norm = Normalizer().fit(raw_data)       # Compute mean & std
X_normalized = norm.transform(raw_data)  # Apply normalization
X_original = norm.inverse(X_normalized)  # Reverse normalization
```

The normalizer parameters are exported alongside the model weights, ensuring consistent inference in the generated Fortran code.

---

## 7. Fiber Directions for Anisotropic Materials

For anisotropic materials, fiber directions can be generated with controlled dispersion:

```python
gen = DeformationGenerator(seed=42)

# Aligned fibers (no dispersion)
fibers = gen.fiber_directions(n=100, mean_direction=np.array([1.0, 0.0, 0.0]))

# Dispersed fibers (half-angle cone of 15°)
fibers_dispersed = gen.fiber_directions(
    n=100,
    mean_direction=np.array([1.0, 0.0, 0.0]),
    dispersion=np.radians(15),
)
```

---

## 8. Visualizing Deformation Data

Use the `Reporter` class to inspect your generated deformations:

```python
from hyper_surrogate.reporting.reporter import Reporter

F = gen.combined(n=5000, stretch_range=(0.8, 1.3))
C = Kinematics.right_cauchy_green(F)

reporter = Reporter(C)

# Individual plots
reporter.fig_invariants()           # I1_bar, I2_bar, J histograms
reporter.fig_principal_stretches()  # λ1, λ2, λ3 distributions
reporter.fig_volume_change()        # J histogram
reporter.fig_eigenvalues()          # Eigenvalue spectra of C

# Full PDF report with all figures
reporter.generate_report("deformation_report/")

# Summary statistics
stats = reporter.basic_statistics()
for key, val in stats.items():
    print(f"{key}: mean={val['mean']:.4f}, std={val['std']:.4f}")
```

---

## 9. Complete Example: Custom Data Pipeline

When you need full control over the data pipeline (e.g. for hybrid UMAT training):

```python
import numpy as np
import hyper_surrogate as hs
from hyper_surrogate.data.dataset import MaterialDataset, Normalizer
from hyper_surrogate.data.deformation import DeformationGenerator
from hyper_surrogate.mechanics.kinematics import Kinematics

# 1. Generate deformations with volumetric perturbation
material = hs.NeoHooke({"C10": 0.5, "KBULK": 1000.0})
n = 20000

gen = DeformationGenerator(seed=42)
F = gen.combined(n, stretch_range=(0.8, 1.3), shear_range=(-0.2, 0.2))

rng = np.random.default_rng(99)
J_target = rng.uniform(0.95, 1.05, size=n)
F = F * (J_target / np.linalg.det(F))[:, None, None] ** (1.0 / 3.0)
C = Kinematics.right_cauchy_green(F)

# 2. Compute inputs (invariants)
i1 = Kinematics.isochoric_invariant1(C)
i2 = Kinematics.isochoric_invariant2(C)
j = np.sqrt(Kinematics.det_invariant(C))
inputs = np.column_stack([i1, i2, j])

# 3. Compute targets (energy + gradient)
energy = material.evaluate_energy(C)
dW_dI = material.evaluate_energy_grad_invariants(C)

# 4. Normalize
in_norm = Normalizer().fit(inputs)
X = in_norm.transform(inputs).astype(np.float32)
W = energy.reshape(-1, 1).astype(np.float32)
S = (dW_dI * in_norm.params["std"]).astype(np.float32)  # Chain rule scaling

# 5. Train/val split
n_val = int(n * 0.15)
idx = np.random.default_rng(42).permutation(n)
train_ds = MaterialDataset(X[idx[n_val:]], (W[idx[n_val:]], S[idx[n_val:]]))
val_ds = MaterialDataset(X[idx[:n_val]], (W[idx[:n_val]], S[idx[:n_val]]))

print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
print(f"Input dim: {X.shape[1]}, Energy range: [{energy.min():.4f}, {energy.max():.4f}]")
```

This custom pipeline gives you full control and is required for `EnergyStressLoss` training (where the target is a `(W, dW/dI)` tuple).
