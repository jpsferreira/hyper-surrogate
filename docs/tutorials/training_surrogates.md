# Training Neural Network Surrogates

This tutorial covers the neural network architectures available in **hyper-surrogate**, the loss functions for thermodynamically consistent training, and the `Trainer` class with early stopping and learning rate scheduling.

---

## Overview

The surrogate training pipeline:

```
Dataset (X, Y)  ──>  Model (MLP / ICNN / PolyconvexICNN / CANN)  ──>  Trainer  ──>  TrainingResult
```

---

## 1. Neural Network Architectures

### Architecture comparison

| Architecture       | Output           |    Convexity     |        Physics Guarantee         | Input Dim | Output Dim |
| ------------------ | ---------------- | :--------------: | :------------------------------: | :-------: | :--------: |
| **MLP**            | Stress or Energy |       None       |               None               |    Any    |    Any     |
| **ICNN**           | Energy (scalar)  | Convex in inputs |    Thermodynamic consistency     |    Any    |     1      |
| **PolyconvexICNN** | Energy (scalar)  |    Polyconvex    | Stronger thermodynamic guarantee |    Any    |     1      |
| **CANN**           | Energy (scalar)  |     Optional     |      Interpretable + sparse      |    Any    |     1      |

### 1.1 MLP (Multi-Layer Perceptron)

The most flexible architecture — a standard feedforward network with no physics constraints:

```python
import hyper_surrogate as hs

model = hs.MLP(
    input_dim=3,          # 3 invariants (I1_bar, I2_bar, J)
    output_dim=6,         # 6 stress components (Voigt)
    hidden_dims=[64, 64], # Two hidden layers with 64 neurons each
    activation="tanh",    # Activation function
)
```

**Architecture diagram:**

```
Input (3) ──> [Linear 64] ──> tanh ──> [Linear 64] ──> tanh ──> [Linear 6] ──> Output (6)
```

**Available activations:**

| Activation   | Function         | Best For                             |
| ------------ | ---------------- | ------------------------------------ |
| `"relu"`     | $\max(0, x)$     | General use, fast                    |
| `"tanh"`     | $\tanh(x)$       | Stress prediction (bounded outputs)  |
| `"sigmoid"`  | $1/(1 + e^{-x})$ | Bounded [0,1] outputs                |
| `"softplus"` | $\ln(1 + e^x)$   | Energy prediction (smooth, positive) |

**When to use MLP:**

- Direct stress prediction (output = PK2 Voigt)
- Quick prototyping without physics constraints
- When convexity is not required

### 1.2 ICNN (Input-Convex Neural Network)

Guarantees that the output is a **convex function** of the input. This is critical for energy-based training: a convex energy function ensures unique stress states and stable FE simulations.

```python
model = hs.ICNN(
    input_dim=3,           # Invariants
    hidden_dims=[32, 32],  # Hidden layer sizes
    activation="softplus", # Must be convex activation
)
# Output dim is always 1 (scalar energy)
```

**Architecture (convexity enforced via non-negative z-path weights):**

```
x ─────────────────────────────────────────────────> Wx₂ ──> + ──> Wx₃ ──> + ──> output
                                                      |       ↑       |       ↑
x ──> [Wx₁] ──> σ ──> z₁ ──> [softplus(Wz₂)]·z₁ ──'  σ  [softplus(Wz₃)]·z₂  σ
                                                            ──> z₂             ──> z₃
```

Key properties:

- **z-path weights** are passed through `softplus()` to enforce non-negativity
- **x-path weights** are unconstrained (direct input skip connections)
- The composition of convex non-decreasing functions with non-negative linear maps preserves convexity

### 1.3 PolyconvexICNN

For **polyconvex** energy functions, each invariant (or group of invariants) gets its own ICNN branch:

```python
# Isotropic: 3 branches, one per invariant
model = hs.PolyconvexICNN(
    groups=[[0], [1], [2]],       # W = W₁(I1_bar) + W₂(I2_bar) + W₃(J)
    hidden_dims=[32, 32],
    activation="softplus",
)

# Anisotropic: 4 branches (fiber invariants grouped)
model = hs.PolyconvexICNN(
    groups=[[0], [1], [2], [3, 4]],  # W = W₁(I1) + W₂(I2) + W₃(J) + W₄(I4, I5)
    hidden_dims=[32, 32],
    activation="softplus",
)
```

**Architecture:**

```
I1_bar ──> [ICNN branch 1] ──> W₁ ──┐
I2_bar ──> [ICNN branch 2] ──> W₂ ──┤
J      ──> [ICNN branch 3] ──> W₃ ──┼──> W = W₁ + W₂ + W₃ (+ W₄)
I4, I5 ──> [ICNN branch 4] ──> W₄ ──┘
```

The sum of convex functions is convex, so polyconvexity is guaranteed by construction.

**Advantages over plain ICNN:**

- Block-diagonal Hessian (more efficient tangent computation)
- Physical interpretability (separate volumetric, isochoric, fiber contributions)
- Better conditioning

### 1.4 CANN (Constitutive Artificial Neural Network)

An interpretable architecture using predefined basis functions with learnable non-negative weights:

```python
from hyper_surrogate.models.cann import CANN

model = CANN(
    input_dim=3,
    n_polynomial=3,          # (I)^1, (I)^2, (I)^3
    n_exponential=2,         # exp(b·I²) - 1 (learnable b)
    use_logarithmic=True,    # log(1 + I²)
    learnable_exponents=True,
)
```

**Energy function:**

$$W = \sum_{k} \text{softplus}(w_k) \cdot \psi_k(I_\alpha)$$

where $\psi_k$ are basis functions applied per invariant:

| Type        | Basis Function $\psi_k$               | Parameters          |
| ----------- | ------------------------------------- | ------------------- |
| Polynomial  | $(I_\alpha)^p,\quad p = 1, \ldots, n$ | None (fixed powers) |
| Exponential | $\exp(b_k I_\alpha^2) - 1$            | Learnable $b_k > 0$ |
| Logarithmic | $\ln(1 + I_\alpha^2)$                 | None                |

**Use case:** Model discovery — after training with L1 regularization (`SparseLoss`), inspect which terms survived to identify the minimal constitutive law.

---

## 2. Loss Functions

### 2.1 StressLoss

Simple MSE on predicted vs. true stress:

$$\mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} \|\mathbf{S}_{\text{pred}}^{(i)} - \mathbf{S}_{\text{true}}^{(i)}\|^2$$

```python
loss_fn = hs.StressLoss()
```

**Use with:** MLP predicting stress directly (`target_type="pk2_voigt"`).

### 2.2 EnergyStressLoss

Joint energy + stress-gradient loss that enforces **thermodynamic consistency**:

$$\mathcal{L} = \alpha \|W_{\text{pred}} - W_{\text{true}}\|^2 + \beta \left\|\frac{\partial W_{\text{pred}}}{\partial \mathbf{x}_{\text{norm}}} - \frac{\partial W_{\text{true}}}{\partial \mathbf{x}_{\text{norm}}}\right\|^2$$

The gradient term is computed via **PyTorch autograd** through the network — this ensures the predicted stress is consistent with the predicted energy.

```python
loss_fn = hs.EnergyStressLoss(alpha=1.0, beta=1.0)
```

| Parameter | Default | Effect                          |
| --------- | :-----: | ------------------------------- |
| `alpha`   |   1.0   | Weight on energy MSE            |
| `beta`    |   1.0   | Weight on gradient (stress) MSE |

**Use with:** ICNN, PolyconvexICNN, MLP (output_dim=1), CANN (`target_type="energy"`).

**Important:** The dataset target must be a tuple `(W, dW/dx_norm)` — see the [data generation tutorial](data_generation.md#9-complete-example-custom-data-pipeline).

### 2.3 StressTangentLoss

Weighted MSE on both stress and material tangent:

$$\mathcal{L} = \alpha \|\mathbf{S}_{\text{pred}} - \mathbf{S}_{\text{true}}\|^2 + \beta \|\mathbb{C}_{\text{pred}} - \mathbb{C}_{\text{true}}\|^2$$

```python
from hyper_surrogate.training.losses import StressTangentLoss

loss_fn = StressTangentLoss(alpha=1.0, beta=0.1)
```

**Use with:** MLP predicting stress + tangent (`target_type="pk2_voigt+cmat_voigt"`, output_dim=27).

### 2.4 SparseLoss

EnergyStressLoss + L1 regularization for model discovery with CANN:

$$\mathcal{L} = \mathcal{L}_{\text{energy+stress}} + \lambda \sum_k |w_k|$$

```python
from hyper_surrogate.training.losses import SparseLoss

loss_fn = SparseLoss(alpha=1.0, beta=1.0, l1_lambda=0.01)
```

**Use with:** CANN architecture for identifying minimal constitutive laws.

### Loss function decision table

| Scenario                     | Loss Function       | Target Type            | Model Output |
| ---------------------------- | ------------------- | ---------------------- | :----------: |
| Direct stress prediction     | `StressLoss`        | `pk2_voigt`            |      6D      |
| Stress + tangent             | `StressTangentLoss` | `pk2_voigt+cmat_voigt` |     27D      |
| Energy-based (thermodynamic) | `EnergyStressLoss`  | `energy`               |      1D      |
| Model discovery (sparse)     | `SparseLoss`        | `energy`               |      1D      |

---

## 3. The Trainer

The `Trainer` class handles the training loop with early stopping and learning rate scheduling:

```python
result = hs.Trainer(
    model,
    train_ds,
    val_ds,
    loss_fn=hs.EnergyStressLoss(alpha=1.0, beta=1.0),
    max_epochs=2000,
    lr=1e-3,
    patience=200,
    batch_size=512,
    device="cpu",  # or "cuda"
).fit()
```

### Trainer parameters

| Parameter    | Type              |    Default     | Description                            |
| ------------ | ----------------- | :------------: | -------------------------------------- |
| `model`      | `SurrogateModel`  |       —        | Network to train                       |
| `train_ds`   | `MaterialDataset` |       —        | Training dataset                       |
| `val_ds`     | `MaterialDataset` |       —        | Validation dataset                     |
| `loss_fn`    | Loss              | `StressLoss()` | Loss function                          |
| `max_epochs` | int               |      1000      | Maximum training epochs                |
| `lr`         | float             |      1e-3      | Initial learning rate (Adam optimizer) |
| `patience`   | int               |      100       | Early stopping patience                |
| `batch_size` | int               |      256       | Mini-batch size                        |
| `device`     | str               |    `"cpu"`     | Device (`"cpu"` or `"cuda"`)           |

### Training features

| Feature                | How It Works                                                         |
| ---------------------- | -------------------------------------------------------------------- |
| **Early stopping**     | Monitors val_loss; stops after `patience` epochs without improvement |
| **LR scheduling**      | `ReduceLROnPlateau`: halves LR when val_loss plateaus                |
| **Best checkpoint**    | Saves best model state; restores it after training completes         |
| **Autograd detection** | Automatically enables gradient computation for `EnergyStressLoss`    |

### TrainingResult

```python
result = trainer.fit()

# Access training history
result.history["train_loss"]  # List[float] per epoch
result.history["val_loss"]    # List[float] per epoch

# Best model
result.model       # The model with best val_loss weights restored
result.best_epoch  # Epoch index of best val_loss

# Print summary
best_val = result.history["val_loss"][result.best_epoch]
print(f"Best val loss: {best_val:.6f} at epoch {result.best_epoch}")
print(f"Total epochs: {len(result.history['train_loss'])}")
```

---

## 4. Example: MLP for Direct Stress Prediction

```python
import hyper_surrogate as hs

material = hs.NeoHooke({"C10": 0.5, "KBULK": 1000.0})

# Create dataset: invariants → PK2 Voigt
train_ds, val_ds, in_norm, out_norm = hs.create_datasets(
    material, n_samples=5000,
    input_type="invariants",
    target_type="pk2_voigt",
    seed=42,
)

# Build model: 3 inputs → 6 outputs
model = hs.MLP(input_dim=3, output_dim=6, hidden_dims=[64, 64], activation="tanh")

# Train
result = hs.Trainer(
    model, train_ds, val_ds,
    loss_fn=hs.StressLoss(),
    max_epochs=500,
    lr=1e-3,
).fit()

print(f"Best val loss: {result.history['val_loss'][result.best_epoch]:.6f}")
```

---

## 5. Example: ICNN for Convex Energy Prediction

```python
import hyper_surrogate as hs

material = hs.NeoHooke({"C10": 0.5, "KBULK": 1000.0})

train_ds, val_ds, in_norm, out_norm = hs.create_datasets(
    material, n_samples=5000,
    input_type="invariants",
    target_type="energy",
    seed=42,
)

# ICNN: convex energy output (scalar)
model = hs.ICNN(input_dim=3, hidden_dims=[32, 32])

# EnergyStressLoss: enforces stress = dW/dI via autograd
result = hs.Trainer(
    model, train_ds, val_ds,
    loss_fn=hs.EnergyStressLoss(alpha=1.0, beta=1.0),
    max_epochs=500,
).fit()

print(f"Best val loss: {result.history['val_loss'][result.best_epoch]:.6f}")
```

---

## 6. Example: PolyconvexICNN for Polyconvex Energy

```python
import hyper_surrogate as hs

material = hs.NeoHooke({"C10": 0.5, "KBULK": 1000.0})

train_ds, val_ds, in_norm, out_norm = hs.create_datasets(
    material, n_samples=5000,
    input_type="invariants",
    target_type="energy",
    seed=42,
)

# Each invariant gets its own convex branch
model = hs.PolyconvexICNN(
    groups=[[0], [1], [2]],  # W = W₁(I1_bar) + W₂(I2_bar) + W₃(J)
    hidden_dims=[32, 32],
    activation="softplus",
)

result = hs.Trainer(
    model, train_ds, val_ds,
    loss_fn=hs.EnergyStressLoss(alpha=1.0, beta=1.0),
    max_epochs=2000,
    lr=1e-3,
    patience=200,
).fit()

print(f"Best val loss: {result.history['val_loss'][result.best_epoch]:.6f}")
```

---

## 7. Example: Energy MLP for Hybrid UMAT

When targeting a hybrid UMAT (NN energy + analytical mechanics), use an MLP with `output_dim=1` and `EnergyStressLoss`:

```python
import numpy as np
import hyper_surrogate as hs
from hyper_surrogate.data.dataset import MaterialDataset, Normalizer
from hyper_surrogate.data.deformation import DeformationGenerator
from hyper_surrogate.mechanics.kinematics import Kinematics

# Generate data with volumetric perturbation
material = hs.NeoHooke({"C10": 0.5, "KBULK": 1000.0})
n = 20000

gen = DeformationGenerator(seed=42)
F = gen.combined(n, stretch_range=(0.8, 1.3), shear_range=(-0.2, 0.2))
rng = np.random.default_rng(99)
J_target = rng.uniform(0.95, 1.05, size=n)
F = F * (J_target / np.linalg.det(F))[:, None, None] ** (1.0 / 3.0)
C = Kinematics.right_cauchy_green(F)

# Inputs and targets
i1 = Kinematics.isochoric_invariant1(C)
i2 = Kinematics.isochoric_invariant2(C)
j = np.sqrt(Kinematics.det_invariant(C))
inputs = np.column_stack([i1, i2, j])

energy = material.evaluate_energy(C)
dW_dI = material.evaluate_energy_grad_invariants(C)

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

# Train energy MLP
model = hs.MLP(input_dim=3, output_dim=1, hidden_dims=[64, 64, 64], activation="softplus")
result = hs.Trainer(
    model, train_ds, val_ds,
    loss_fn=hs.EnergyStressLoss(alpha=1.0, beta=1.0),
    max_epochs=3000, lr=1e-3, patience=300, batch_size=512,
).fit()

best_val = result.history["val_loss"][result.best_epoch]
print(f"Best val loss: {best_val:.6f} (epoch {result.best_epoch})")
```

---

## 8. Evaluating a Trained Model

### Direct inference

```python
import torch

model.eval()
x_test = torch.tensor(val_ds.inputs[:100], dtype=torch.float32)

with torch.no_grad():
    y_pred = model(x_test).numpy()
```

### Hybrid inference (energy → stress via autograd)

For energy-based models, compute stress as the gradient of energy:

```python
model.eval()
x = torch.tensor(val_ds.inputs[:100], dtype=torch.float32, requires_grad=True)

W_pred = model(x)
dW_dx = torch.autograd.grad(W_pred.sum(), x, create_graph=True)[0]

# Convert from normalized to raw invariant space
std_t = torch.tensor(in_norm.params["std"], dtype=torch.float32)
dW_dI_pred = (dW_dx / std_t).detach().numpy()  # (100, 3)
```

### Computing R² and MAE

```python
import numpy as np

# Compare predictions to reference
dW_dI_ref = dW_dI[idx[:n_val]][:100]

for i, name in enumerate(["dW/dI1_bar", "dW/dI2_bar", "dW/dJ"]):
    ss_res = np.sum((dW_dI_pred[:, i] - dW_dI_ref[:, i]) ** 2)
    ss_tot = np.sum((dW_dI_ref[:, i] - dW_dI_ref[:, i].mean()) ** 2)
    r2 = 1 - ss_res / max(ss_tot, 1e-30)
    mae = np.mean(np.abs(dW_dI_pred[:, i] - dW_dI_ref[:, i]))
    print(f"  {name}: R²={r2:.6f}, MAE={mae:.6f}")
```

---

## 9. Hyperparameter Guidelines

### Recommended starting configurations

| Scenario          | Architecture   |  Hidden Dims   | Activation | Loss               | Epochs |  LR  | Patience |
| ----------------- | -------------- | :------------: | :--------: | ------------------ | :----: | :--: | :------: |
| Quick prototype   | MLP            |   `[32, 32]`   |   `tanh`   | `StressLoss`       |  500   | 1e-3 |    50    |
| Production stress | MLP            | `[64, 64, 64]` |   `tanh`   | `StressLoss`       |  2000  | 1e-3 |   200    |
| Convex energy     | ICNN           |   `[32, 32]`   | `softplus` | `EnergyStressLoss` |  1000  | 1e-3 |   100    |
| Polyconvex energy | PolyconvexICNN |   `[32, 32]`   | `softplus` | `EnergyStressLoss` |  2000  | 1e-3 |   200    |
| Hybrid UMAT       | MLP (out=1)    | `[64, 64, 64]` | `softplus` | `EnergyStressLoss` |  3000  | 1e-3 |   300    |
| Model discovery   | CANN           |       —        |     —      | `SparseLoss`       |  500   | 1e-3 |    50    |

### Tips

- **Activation**: Use `softplus` for energy models (smooth, positive), `tanh` for stress models
- **Batch size**: 256--512 works well; larger batches stabilize gradients
- **Learning rate**: Start at 1e-3; the scheduler will reduce automatically
- **Data size**: 5000--20000 samples is usually sufficient for 3D invariants
- **Hidden dimensions**: 32--64 neurons per layer, 2--3 layers is typical
