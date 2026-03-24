# Anisotropic Materials

This tutorial covers modeling fiber-reinforced biological tissues — from single-fiber arterial models to multi-fiber cardiac tissue — including data generation, training, and Fortran export for anisotropic materials.

---

## Overview

Anisotropic hyperelastic materials have a **preferred direction** (fiber) that makes the mechanical response direction-dependent. In `hyper-surrogate`, anisotropy is introduced via fiber (pseudo-)invariants:

| Invariant | Expression | Physical Meaning |
|-----------|-----------|-----------------|
| $I_4$ | $\mathbf{a}_0 \cdot \mathbf{C}\, \mathbf{a}_0$ | Squared fiber stretch ($\lambda_f^2$) |
| $I_5$ | $\mathbf{a}_0 \cdot \mathbf{C}^2\, \mathbf{a}_0$ | Fiber-shear coupling |

The NN input becomes **5-dimensional**: $[\bar{I}_1, \bar{I}_2, J, I_4, I_5]$ instead of 3D for isotropic materials.

---

## 1. Holzapfel-Ogden: Single Fiber Family

### 1.1 Model definition

```python
import numpy as np
import hyper_surrogate as hs

fiber_dir = np.array([1.0, 0.0, 0.0])  # Fiber along x-axis

material = hs.HolzapfelOgden(
    parameters={
        "a": 0.059,     # Ground substance stiffness (kPa)
        "b": 8.023,     # Ground substance exponent
        "af": 18.472,   # Fiber stiffness (kPa)
        "bf": 16.026,   # Fiber exponent
        "KBULK": 1000.0,
    },
    fiber_direction=fiber_dir,
)
```

### 1.2 Evaluating stress

```python
from hyper_surrogate.mechanics.kinematics import Kinematics

# Uniaxial stretch along fiber direction
lam = 1.15
F = np.array([[[lam, 0, 0],
               [0, 1.0/np.sqrt(lam), 0],
               [0, 0, 1.0/np.sqrt(lam)]]])

C = Kinematics.right_cauchy_green(F)
pk2 = material.evaluate_pk2(C)       # (1, 3, 3)
energy = material.evaluate_energy(C)  # (1,)

print(f"Energy: {energy[0]:.6f}")
print(f"PK2 S11 (fiber dir): {pk2[0, 0, 0]:.6f}")
print(f"PK2 S22 (transverse): {pk2[0, 1, 1]:.6f}")
```

Note the **anisotropy**: $S_{11} \gg S_{22}$ because of the fiber contribution along axis 1.

### 1.3 Computing fiber invariants

```python
gen = hs.DeformationGenerator(seed=42)
F_batch = gen.combined(n=5000, stretch_range=(0.8, 1.3))
C_batch = Kinematics.right_cauchy_green(F_batch)

# Isochoric invariants
i1 = Kinematics.isochoric_invariant1(C_batch)
i2 = Kinematics.isochoric_invariant2(C_batch)
j = np.sqrt(Kinematics.det_invariant(C_batch))

# Fiber invariants
i4 = Kinematics.fiber_invariant4(C_batch, fiber_dir)
i5 = Kinematics.fiber_invariant5(C_batch, fiber_dir)

# 5D input for NN
inputs = np.column_stack([i1, i2, j, i4, i5])
print(f"Input shape: {inputs.shape}")  # (5000, 5)
print(f"I4 range: [{i4.min():.3f}, {i4.max():.3f}]")
print(f"I5 range: [{i5.min():.3f}, {i5.max():.3f}]")
```

### 1.4 Full training pipeline

```python
from hyper_surrogate.data.dataset import MaterialDataset, Normalizer

n = 10000
F = gen.combined(n, stretch_range=(0.8, 1.3), shear_range=(-0.2, 0.2))

# Add volumetric perturbation
rng = np.random.default_rng(99)
J_target = rng.uniform(0.95, 1.05, size=n)
F = F * (J_target / np.linalg.det(F))[:, None, None] ** (1.0 / 3.0)
C = Kinematics.right_cauchy_green(F)

# 5D invariants
i1 = Kinematics.isochoric_invariant1(C)
i2 = Kinematics.isochoric_invariant2(C)
j = np.sqrt(Kinematics.det_invariant(C))
i4 = Kinematics.fiber_invariant4(C, fiber_dir)
i5 = Kinematics.fiber_invariant5(C, fiber_dir)
inputs = np.column_stack([i1, i2, j, i4, i5])

# Energy and gradient targets (5 components)
energy = material.evaluate_energy(C)
dW_dI = material.evaluate_energy_grad_invariants(C)  # (N, 5)

# Normalize
in_norm = Normalizer().fit(inputs)
X = in_norm.transform(inputs).astype(np.float32)
W = energy.reshape(-1, 1).astype(np.float32)
S = (dW_dI * in_norm.params["std"]).astype(np.float32)

# Split
n_val = int(n * 0.15)
idx = np.random.default_rng(42).permutation(n)
train_ds = MaterialDataset(X[idx[n_val:]], (W[idx[n_val:]], S[idx[n_val:]]))
val_ds = MaterialDataset(X[idx[:n_val]], (W[idx[:n_val]], S[idx[:n_val]]))

# Train with 5 inputs
model = hs.MLP(input_dim=5, output_dim=1, hidden_dims=[64, 64, 64], activation="softplus")
result = hs.Trainer(
    model, train_ds, val_ds,
    loss_fn=hs.EnergyStressLoss(alpha=1.0, beta=1.0),
    max_epochs=3000, lr=1e-3, patience=300, batch_size=512,
).fit()

print(f"Best val loss: {result.history['val_loss'][result.best_epoch]:.6f}")

# Export hybrid UMAT (auto-detects 5D → anisotropic)
energy_norm = Normalizer().fit(energy.reshape(-1, 1))
exported = hs.extract_weights(result.model, in_norm, energy_norm)
hs.HybridUMATEmitter(exported).write("holzapfel_ogden_umat.f90")
```

The `HybridUMATEmitter` detects `input_dim=5` and generates Fortran code that:

- Reads the fiber direction from `PROPS(1:3)`
- Computes $I_4, I_5$ from $\mathbf{C}$ and $\mathbf{a}_0$
- Includes $\partial I_4 / \partial \mathbf{C}$ and $\partial I_5 / \partial \mathbf{C}$ for the stress chain rule

---

## 2. GOH: Dispersed Fibers

The Gasser-Ogden-Holzapfel model adds a **dispersion parameter** $\kappa$ that blends fiber and isotropic contributions:

$$\bar{E} = \kappa(\bar{I}_1 - 3) + (1 - 3\kappa)(I_4 - 1)$$

| $\kappa$ | Interpretation |
|:--------:|----------------|
| $0$ | Perfectly aligned (= Holzapfel-Ogden) |
| $0.1$ | Slightly dispersed |
| $0.226$ | Moderate dispersion (typical arterial media) |
| $1/3$ | Fully isotropic (no preferred direction) |

```python
from hyper_surrogate.mechanics.materials import GasserOgdenHolzapfel

mat_goh = GasserOgdenHolzapfel(
    parameters={
        "a": 0.059, "b": 8.023,
        "af": 18.472, "bf": 16.026,
        "kappa": 0.226,
        "KBULK": 1000.0,
    },
    fiber_direction=np.array([1.0, 0.0, 0.0]),
)

# Same workflow: evaluate_pk2, evaluate_energy, etc.
pk2 = mat_goh.evaluate_pk2(C)
energy = mat_goh.evaluate_energy(C)
```

---

## 3. Two-Fiber Arterial Wall Model

Real arteries have **two families** of collagen fibers oriented symmetrically at angles $\pm\theta$ from the circumferential direction.

### 3.1 Setup

```python
import numpy as np
from hyper_surrogate.mechanics.materials import GasserOgdenHolzapfel
from hyper_surrogate.mechanics.kinematics import Kinematics

# Two fiber families at ±39° from circumferential (x-axis)
theta = np.radians(39.0)
fiber1 = np.array([np.cos(theta), np.sin(theta), 0.0])
fiber2 = np.array([np.cos(theta), -np.sin(theta), 0.0])

# Medial layer parameters (Holzapfel et al., 2005)
params = {
    "a": 3.0,       # kPa
    "b": 0.5,
    "af": 2.3632,   # kPa
    "bf": 0.8393,
    "kappa": 0.226,
    "KBULK": 100.0,
}

mat1 = GasserOgdenHolzapfel(parameters=params, fiber_direction=fiber1)
mat2 = GasserOgdenHolzapfel(parameters=params, fiber_direction=fiber2)
```

### 3.2 Combining two fiber families

The total energy avoids double-counting the isotropic ground substance:

$$W_{\text{total}} = \underbrace{W_{\text{iso}} + W_{\text{vol}}}_{\text{shared (from fiber 1)}} + \underbrace{W_{\text{fiber,1}}}_{\text{fiber 1}} + \underbrace{W_{\text{fiber,2}}}_{\text{fiber 2}}$$

```python
from sympy import Symbol, lambdify

# Symbolic invariants
i1s, i2s, js = Symbol("I1b"), Symbol("I2b"), Symbol("J")
i4s_1, i5s_1 = Symbol("I4_1"), Symbol("I5_1")
i4s_2, i5s_2 = Symbol("I4_2"), Symbol("I5_2")

# Full energy from fiber family 1 (isotropic + volumetric + fiber)
W1_full = mat1.sef_from_invariants(i1s, i2s, js, i4s_1, i5s_1)

# Fiber-only contribution from family 2
W2_full = mat2.sef_from_invariants(i1s, i2s, js, i4s_2, i5s_2)
W2_iso = mat2.sef_from_invariants(i1s, i2s, js)  # iso + vol only
W2_fiber = W2_full - W2_iso  # fiber contribution only

W_total = W1_full + W2_fiber
```

### 3.3 Evaluating under equibiaxial stretch

```python
stretches = np.linspace(1.0, 1.3, 10)

print(f"{'Stretch':>8s} {'W_total':>12s} {'I4_fib1':>10s} {'I4_fib2':>10s}")
print("-" * 45)

for lam in stretches:
    F = np.array([[[lam, 0, 0], [0, lam, 0], [0, 0, 1.0/(lam*lam)]]])
    C = Kinematics.right_cauchy_green(F)

    i4_1 = Kinematics.fiber_invariant4(C, fiber1)
    i4_2 = Kinematics.fiber_invariant4(C, fiber2)

    W1 = mat1.evaluate_energy(C)
    W2 = mat2.evaluate_energy(C)
    # Total = W1 (includes iso+vol) + fiber-only from mat2
    # For approximate evaluation, sum both energies and subtract one iso+vol contribution

    print(f"{lam:8.3f} {float(W1 + W2):12.4f} {float(i4_1):10.4f} {float(i4_2):10.4f}")
```

### 3.4 Important notes for multi-fiber models

| Aspect | Guideline |
|--------|-----------|
| **Double-counting** | The isotropic + volumetric parts must only be counted once |
| **Symmetry** | For symmetric fiber families, $W_{\text{fiber,1}}(\theta) = W_{\text{fiber,2}}(-\theta)$ under symmetric loading |
| **Macaulay bracket** | Fibers only contribute under tension ($I_4 > 1$) |
| **NN approach** | Train separate NNs per fiber family, or one NN with all fiber invariants as inputs |

---

## 4. Guccione: Cardiac Tissue

The Guccione model requires **two direction vectors** — fiber direction and sheet direction:

```python
from hyper_surrogate.mechanics.materials import Guccione

mat = Guccione(
    parameters={
        "C": 0.876,       # Overall stiffness (kPa)
        "bf": 18.48,      # Fiber direction
        "bt": 3.58,       # Transverse (sheet/normal)
        "bfs": 1.627,     # Fiber-sheet shear
        "KBULK": 1000.0,
    },
    fiber_direction=np.array([1.0, 0.0, 0.0]),  # f₀
    sheet_direction=np.array([0.0, 1.0, 0.0]),   # s₀
)
# Normal direction n₀ = f₀ × s₀ is computed automatically
```

### Evaluating Guccione

```python
# Stretch along fiber direction
lam = 1.1
F = np.array([[[lam, 0, 0],
               [0, 1.0/np.sqrt(lam), 0],
               [0, 0, 1.0/np.sqrt(lam)]]])

C = Kinematics.right_cauchy_green(F)
pk2 = mat.evaluate_pk2(C)
energy = mat.evaluate_energy(C)

print(f"Energy: {energy[0]:.6f}")
print(f"PK2 S11 (fiber): {pk2[0, 0, 0]:.6f}")
print(f"PK2 S22 (sheet): {pk2[0, 1, 1]:.6f}")
print(f"PK2 S33 (normal): {pk2[0, 2, 2]:.6f}")
```

### Physical interpretation of Guccione parameters

| Parameter | Large value means... |
|-----------|---------------------|
| $b_f$ | Stiffer in fiber direction |
| $b_t$ | Stiffer transversely (sheet & normal) |
| $b_{fs}$ | Stiffer in fiber-sheet shear |

Typical hierarchy: $b_f > b_t > b_{fs}$ (myocardium is stiffest along fibers).

---

## 5. Anisotropic Data Generation Tips

### 5.1 Fiber-aligned deformations

Ensure your training data covers fiber-relevant deformation states:

```python
gen = hs.DeformationGenerator(seed=42)

# Combined deformations include rotations that naturally probe fiber directions
F = gen.combined(n=10000, stretch_range=(0.8, 1.3), shear_range=(-0.2, 0.2))
```

### 5.2 Check fiber invariant coverage

```python
C = Kinematics.right_cauchy_green(F)
i4 = Kinematics.fiber_invariant4(C, fiber_dir)

print(f"I4 range: [{i4.min():.3f}, {i4.max():.3f}]")
print(f"I4 < 1 (compression): {(i4 < 1).sum()} / {len(i4)}")
print(f"I4 > 1 (tension): {(i4 > 1).sum()} / {len(i4)}")

# For HolzapfelOgden, fibers only activate under tension (I4 > 1)
# Ensure sufficient tension samples
```

### 5.3 Dispersed fiber directions

For materials with fiber dispersion, you can generate varied fiber orientations:

```python
fibers = gen.fiber_directions(
    n=100,
    mean_direction=np.array([1.0, 0.0, 0.0]),
    dispersion=np.radians(15),  # 15° cone half-angle
)
# fibers.shape = (100, 3), all unit vectors
```

---

## 6. Summary: Anisotropic Model Comparison

| Model | Fiber Families | Directions Needed | Key Feature | Application |
|-------|:--------------:|:-----------------:|-------------|-------------|
| **HolzapfelOgden** | 1 | `fiber_direction` | Macaulay bracket (tension-only) | Arterial media |
| **GOH** | 1 | `fiber_direction` | Dispersion parameter $\kappa$ | Arterial adventitia |
| **Guccione** | 1 | `fiber_direction` + `sheet_direction` | Fiber-sheet-normal frame | Myocardium |
| **Two-fiber** | 2 | Two `fiber_direction`s | Combined energy (avoid double-counting) | Full arterial wall |

### NN input dimensions

| Model Type | Input Dim | Components |
|-----------|:---------:|-----------|
| Isotropic | 3 | $\bar{I}_1, \bar{I}_2, J$ |
| Single fiber | 5 | $\bar{I}_1, \bar{I}_2, J, I_4, I_5$ |
| Two fibers | 7 | $\bar{I}_1, \bar{I}_2, J, I_4^{(1)}, I_5^{(1)}, I_4^{(2)}, I_5^{(2)}$ |
