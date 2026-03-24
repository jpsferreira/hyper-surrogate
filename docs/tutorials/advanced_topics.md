# Advanced Topics

This tutorial covers model discovery with CANN, parameter fitting to experimental data, benchmarking surrogates, and generating deformation diagnostic reports.

---

## 1. Model Discovery with CANN

The **Constitutive Artificial Neural Network (CANN)** uses interpretable basis functions with learnable non-negative weights. Combined with L1 regularization, it discovers the **minimal constitutive law** from data.

### 1.1 The idea

Instead of a black-box NN, CANN expresses the energy as:

$$W = \sum_{k} \underbrace{\text{softplus}(w_k)}_{\ge 0} \cdot \psi_k(I_\alpha)$$

After training with sparsity regularization, most weights $w_k \to 0$. The surviving terms reveal the underlying constitutive law.

### 1.2 Setting up a CANN

```python
from hyper_surrogate.models.cann import CANN
from hyper_surrogate.data.dataset import create_datasets
from hyper_surrogate.mechanics.materials import NeoHooke

# Generate data from a known material (ground truth)
material = NeoHooke(parameters={"C10": 0.5, "KBULK": 1000.0})
train_ds, val_ds, in_norm, out_norm = create_datasets(
    material, n_samples=5000,
    input_type="invariants", target_type="energy", seed=42,
)

# Create CANN with diverse basis functions
model = CANN(
    input_dim=train_ds.inputs.shape[1],  # 3 (I1_bar, I2_bar, J)
    n_polynomial=3,       # (I)^1, (I)^2, (I)^3 per invariant
    n_exponential=2,      # exp(b·I²) - 1 per invariant (learnable b)
    use_logarithmic=True, # log(1 + I²) per invariant
    learnable_exponents=True,
)

print(f"Total basis functions: {model._n_basis}")
```

### 1.3 Training with sparsity

```python
from hyper_surrogate.training.losses import SparseLoss
from hyper_surrogate.training.trainer import Trainer

# SparseLoss = EnergyStressLoss + L1 regularization
loss_fn = SparseLoss(alpha=1.0, beta=1.0, l1_lambda=0.01)

result = Trainer(
    model, train_ds, val_ds,
    loss_fn=loss_fn,
    max_epochs=500,
    lr=1e-3,
).fit()

print(f"Final val loss: {result.history['val_loss'][-1]:.6f}")
```

### 1.4 Inspecting discovered terms

```python
# Get active (non-zero) basis functions
terms = model.get_active_terms(threshold=0.01)

print("Discovered constitutive law:")
for t in terms:
    inv_name = f"I{t['invariant'] + 1}_bar" if t["invariant"] < 2 else "J"
    w = t["weight"]

    if t["type"] == "polynomial":
        print(f"  w={w:.4f} * ({inv_name})^{t['power']}")
    elif t["type"] == "exponential":
        print(f"  w={w:.4f} * [exp({t['stiffness']:.3f} * {inv_name}^2) - 1]")
    elif t["type"] == "logarithmic":
        print(f"  w={w:.4f} * log(1 + {inv_name}^2)")

print(f"\nActive terms: {len(terms)} / {model._n_basis}")
```

**Expected output for NeoHooke** ($W = C_{10}(\bar{I}_1 - 3)$): the dominant term should be a linear polynomial in $\bar{I}_1$.

### 1.5 CANN basis functions reference

| Type               | Function $\psi_k(I)$    | Learnable Parameters |
| ------------------ | ----------------------- | -------------------- |
| Polynomial ($p=1$) | $I$                     | None                 |
| Polynomial ($p=2$) | $I^2$                   | None                 |
| Polynomial ($p=3$) | $I^3$                   | None                 |
| Exponential        | $\exp(b \cdot I^2) - 1$ | $b > 0$              |
| Logarithmic        | $\ln(1 + I^2)$          | None                 |

Each basis function is applied **per invariant**, so with 3 invariants and 6 basis types, there are 18 total terms.

---

## 2. Parameter Fitting to Experimental Data

### 2.1 Loading experimental data

`hyper-surrogate` includes the classic **Treloar (1944)** rubber dataset:

```python
from hyper_surrogate.data.experimental import ExperimentalData

# Load built-in reference dataset
data = ExperimentalData.load_reference("treloar")

print(f"Data points: {len(data.stretch)}")
print(f"Stretch range: [{data.stretch.min():.2f}, {data.stretch.max():.2f}]")
print(f"Stress range: [{data.stress.min():.3f}, {data.stress.max():.3f}] MPa")
```

The `ExperimentalData` object contains:

| Field       | Type      | Description                              |
| ----------- | --------- | ---------------------------------------- |
| `stretch`   | `ndarray` | Principal stretch values $\lambda$       |
| `stress`    | `ndarray` | Corresponding engineering stress (MPa)   |
| `test_type` | `str`     | Type of test (`"uniaxial"`, `"biaxial"`) |

### 2.2 Fitting material parameters

```python
from hyper_surrogate.data.fitting import fit_material
from hyper_surrogate.mechanics.materials import NeoHooke, MooneyRivlin, Yeoh

# Fit Neo-Hooke
mat_nh, res_nh = fit_material(
    NeoHooke,
    data,
    initial_guess={"C10": 0.1},
    fixed_params={"KBULK": 1000.0},
)
print(f"NeoHooke: C10={res_nh.parameters['C10']:.4f}, R²={res_nh.r_squared:.4f}")

# Fit Mooney-Rivlin
mat_mr, res_mr = fit_material(
    MooneyRivlin,
    data,
    initial_guess={"C10": 0.1, "C01": 0.05},
    fixed_params={"KBULK": 1000.0},
)
print(f"MooneyRivlin: C10={res_mr.parameters['C10']:.4f}, "
      f"C01={res_mr.parameters['C01']:.4f}, R²={res_mr.r_squared:.4f}")

# Fit Yeoh
mat_ye, res_ye = fit_material(
    Yeoh,
    data,
    initial_guess={"C10": 0.1, "C20": 0.001, "C30": 0.0001},
    fixed_params={"KBULK": 1000.0},
)
print(f"Yeoh: C10={res_ye.parameters['C10']:.4f}, "
      f"C20={res_ye.parameters['C20']:.6f}, "
      f"C30={res_ye.parameters['C30']:.8f}, R²={res_ye.r_squared:.4f}")
```

### 2.3 Understanding the fitting result

The `fit_material` function returns a `(Material, FitResult)` tuple:

| `FitResult` Field | Description                                  |
| ----------------- | -------------------------------------------- |
| `parameters`      | `Dict[str, float]` — fitted parameter values |
| `r_squared`       | $R^2$ goodness-of-fit (1.0 = perfect)        |
| `residual`        | Sum of squared stress residuals              |

The fitting minimizes:

$$\min_{\theta} \sum_i \|\sigma_{\text{model}}(\lambda_i; \theta) - \sigma_{\text{exp},i}\|^2$$

using `scipy.optimize.minimize`.

### 2.4 Comparing model fits

| Model          | Parameters |   Expected $R^2$ (Treloar)    |
| -------------- | :--------: | :---------------------------: |
| NeoHooke       |     1      | ~0.95 (poor at large stretch) |
| MooneyRivlin   |     2      |             ~0.97             |
| Yeoh           |     3      | ~0.99+ (captures stiffening)  |
| Ogden (3-term) |     6      |            ~0.999             |

### 2.5 Using fitted material for surrogate training

```python
# Use the fitted material directly in the surrogate pipeline
train_ds, val_ds, in_norm, out_norm = hs.create_datasets(
    mat_ye,  # Fitted Yeoh material
    n_samples=10000,
    input_type="invariants",
    target_type="energy",
)
# ... train NN as usual ...
```

---

## 3. Benchmarking Surrogates

### 3.1 Using the benchmark suite

The `benchmark_suite` function systematically compares architectures across materials:

```python
from hyper_surrogate.benchmarking.metrics import benchmark_suite
from hyper_surrogate.benchmarking.reporting import results_to_markdown
from hyper_surrogate.mechanics.materials import NeoHooke, MooneyRivlin, Yeoh
from hyper_surrogate.models.mlp import MLP
from hyper_surrogate.models.icnn import ICNN
from hyper_surrogate.training.losses import StressLoss

materials = [NeoHooke(), MooneyRivlin(), Yeoh()]

model_configs = [
    {
        "name": "MLP-64x64",
        "model_cls": MLP,
        "kwargs": {"hidden_dims": [64, 64], "activation": "tanh"},
        "loss_cls": StressLoss,
        "epochs": 200,
        "target_type": "pk2_voigt",
    },
    {
        "name": "ICNN-64x64",
        "model_cls": ICNN,
        "kwargs": {"hidden_dims": [64, 64]},
        "loss_cls": StressLoss,
        "epochs": 200,
        "target_type": "pk2_voigt",
    },
]

results = benchmark_suite(materials, model_configs, n_samples=5000, seed=42)

# Pretty-print as markdown table
print(results_to_markdown(results))

# Per-result details
for r in results:
    print(r.summary())
```

### 3.2 Benchmark metrics

Each `BenchmarkResult` contains:

| Metric        | Description                              |
| ------------- | ---------------------------------------- |
| Material name | Which material was tested                |
| Model name    | Which architecture was used              |
| $R^2$         | Coefficient of determination on test set |
| MAE           | Mean absolute error                      |
| RMSE          | Root mean squared error                  |
| Training time | Wall-clock time for training             |
| Best epoch    | Epoch with best validation loss          |

### 3.3 Example output

```
| Material     | Model      | R²      | MAE     | RMSE    | Time (s) |
|-------------|------------|---------|---------|---------|----------|
| NeoHooke    | MLP-64x64  | 0.9998  | 0.0012  | 0.0018  | 12.3     |
| NeoHooke    | ICNN-64x64 | 0.9995  | 0.0019  | 0.0025  | 15.1     |
| MooneyRivlin| MLP-64x64  | 0.9997  | 0.0015  | 0.0021  | 13.0     |
| MooneyRivlin| ICNN-64x64 | 0.9993  | 0.0022  | 0.0030  | 16.2     |
| Yeoh        | MLP-64x64  | 0.9996  | 0.0018  | 0.0024  | 13.5     |
| Yeoh        | ICNN-64x64 | 0.9991  | 0.0026  | 0.0035  | 17.0     |
```

---

## 4. Deformation Reporting

### 4.1 Generating diagnostic reports

The `Reporter` class creates visualizations of your deformation data to verify coverage and quality:

```python
import numpy as np
from hyper_surrogate.data.deformation import DeformationGenerator
from hyper_surrogate.mechanics.kinematics import Kinematics
from hyper_surrogate.reporting.reporter import Reporter

# Generate data
gen = DeformationGenerator(seed=42)
F = gen.combined(n=5000, stretch_range=(0.8, 1.3), shear_range=(-0.2, 0.2))
C = Kinematics.right_cauchy_green(F)

# Create reporter
reporter = Reporter(C)
```

### 4.2 Individual plots

```python
# Eigenvalue distribution of C (per component)
reporter.fig_eigenvalues()

# Determinant det(C) = J² histogram
reporter.fig_determinants()

# Isochoric invariants: I1_bar, I2_bar, J
reporter.fig_invariants()

# Principal stretches λ1 ≥ λ2 ≥ λ3
reporter.fig_principal_stretches()

# Volume change ratio J
reporter.fig_volume_change()
```

### 4.3 Summary statistics

```python
stats = reporter.basic_statistics()
for key, val in stats.items():
    print(f"{key:>20s}: mean={val['mean']:8.4f}, std={val['std']:8.4f}, "
          f"min={val['min']:8.4f}, max={val['max']:8.4f}")
```

Example output:

```
           I1_bar:  mean=  3.0842, std=  0.1523, min=  2.7601, max=  3.9214
           I2_bar:  mean=  3.1056, std=  0.1891, min=  2.7012, max=  4.2103
               J:  mean=  1.0000, std=  0.0001, min=  0.9998, max=  1.0002
         lambda1:  mean=  1.1203, std=  0.0812, min=  0.8521, max=  1.4012
         lambda2:  mean=  1.0012, std=  0.0534, min=  0.8102, max=  1.2503
         lambda3:  mean=  0.8956, std=  0.0612, min=  0.7201, max=  1.1203
```

### 4.4 Full PDF report

```python
# Generate all figures as a combined report
reporter.generate_report("deformation_report/")
# Creates: deformation_report/ with all figures as PNG + summary
```

### 4.5 From deformation gradients

You can also pass $\mathbf{F}$ directly:

```python
reporter = Reporter(F, tensor_type="F")  # Computes C = F^T F internally
reporter.generate_report("report_from_F/")
```

### 4.6 What to look for in diagnostic reports

| Check                          | What It Tells You                                                            |
| ------------------------------ | ---------------------------------------------------------------------------- |
| $\bar{I}_1$ near 3             | Deformations are moderate (good for most materials)                          |
| $J$ spread                     | Volumetric perturbation range (should match your material's compressibility) |
| $\lambda_1 / \lambda_3$ ratio  | Maximum stretch anisotropy                                                   |
| Bimodal $\lambda$ distribution | Mix of tension and compression modes                                         |
| Uniform $J$ histogram          | Even volumetric sampling                                                     |

---

## 5. Workflow Recipes

### Recipe: From experimental data to Fortran UMAT

```python
import hyper_surrogate as hs
from hyper_surrogate.data.experimental import ExperimentalData
from hyper_surrogate.data.fitting import fit_material
from hyper_surrogate.mechanics.materials import Yeoh

# 1. Load and fit
data = ExperimentalData.load_reference("treloar")
material, fit_result = fit_material(
    Yeoh, data,
    initial_guess={"C10": 0.1, "C20": 0.001, "C30": 0.0001},
    fixed_params={"KBULK": 1000.0},
)
print(f"Fit R² = {fit_result.r_squared:.4f}")

# 2. Generate surrogate training data
train_ds, val_ds, in_norm, out_norm = hs.create_datasets(
    material, n_samples=10000,
    input_type="invariants", target_type="energy",
)

# 3. Train
model = hs.ICNN(input_dim=3, hidden_dims=[64, 64])
result = hs.Trainer(
    model, train_ds, val_ds,
    loss_fn=hs.EnergyStressLoss(alpha=1.0, beta=1.0),
    max_epochs=2000, lr=1e-3, patience=200,
).fit()

# 4. Export
exported = hs.extract_weights(result.model, in_norm, out_norm)
hs.HybridUMATEmitter(exported).write("yeoh_surrogate_umat.f90")
```

### Recipe: Analytical UMAT (no NN)

```python
from hyper_surrogate.mechanics.materials import MooneyRivlin
from hyper_surrogate.export.fortran.analytical import UMATHandler

material = MooneyRivlin({"C10": 0.3, "C01": 0.2, "KBULK": 1000.0})
UMATHandler(material).generate("mooneyrivlin_umat.f")
```

### Recipe: Model discovery pipeline

```python
from hyper_surrogate.models.cann import CANN
from hyper_surrogate.training.losses import SparseLoss

# 1. Train CANN with sparsity
model = CANN(input_dim=3, n_polynomial=3, n_exponential=2, use_logarithmic=True)
result = hs.Trainer(
    model, train_ds, val_ds,
    loss_fn=SparseLoss(alpha=1.0, beta=1.0, l1_lambda=0.01),
    max_epochs=500,
).fit()

# 2. Identify active terms
terms = model.get_active_terms(threshold=0.01)
print(f"Discovered {len(terms)} active terms out of {model._n_basis}")

# 3. Use discovered structure to select a simpler analytical model
# e.g., if only I1_bar polynomial terms survive → NeoHooke or Yeoh
```
