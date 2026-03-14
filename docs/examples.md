# Examples

## Quick start: Evaluate stress for a Neo-Hookean material

```python
import numpy as np
import hyper_surrogate as hs

# Define a Neo-Hookean material
material = hs.NeoHooke({"C10": 0.5, "KBULK": 1000.0})

# Generate uniaxial deformation gradients
gen = hs.DeformationGenerator(seed=42)
F = gen.uniaxial(n=10)

# Compute the right Cauchy-Green tensor
C = hs.Kinematics.right_cauchy_green(F)

# Evaluate PK2 stress (N, 3, 3) and strain energy (N,)
pk2 = material.evaluate_pk2(C)
energy = material.evaluate_energy(C)

print(f"PK2 stress shape: {pk2.shape}")
print(f"Energy shape: {energy.shape}")
```

## Generate an analytical Fortran UMAT

Generate a Fortran 90 subroutine for a Neo-Hookean material with common subexpression elimination:

```python
from hyper_surrogate.mechanics.materials import NeoHooke
from hyper_surrogate.export.fortran.analytical import UMATHandler

material = NeoHooke({"C10": 0.5, "KBULK": 1000.0})
handler = UMATHandler(material)
handler.generate("neohooke_umat.f")
```

This writes a complete UMAT subroutine with Cauchy stress and spatial tangent stiffness, ready for use in Abaqus or LS-DYNA.

## Train an MLP surrogate and export to Fortran

Full pipeline: generate data, train a neural network, and export to a standalone Fortran 90 module.

```python
import hyper_surrogate as hs

# 1. Define material and generate training data
material = hs.NeoHooke({"C10": 0.5, "KBULK": 1000.0})
train_ds, val_ds, in_norm, out_norm = hs.create_datasets(
    material,
    n_samples=5000,
    input_type="invariants",   # 3 inputs: I1_bar, I2_bar, J
    target_type="pk2_voigt",   # 6 outputs: PK2 stress in Voigt notation
)

# 2. Build and train the model
model = hs.MLP(input_dim=3, output_dim=6, hidden_dims=[32, 32], activation="tanh")
result = hs.Trainer(
    model, train_ds, val_ds,
    loss_fn=hs.StressLoss(),
    max_epochs=500,
    lr=1e-3,
).fit()

best_val = result.history["val_loss"][result.best_epoch]
print(f"Best validation loss: {best_val:.6f} (epoch {result.best_epoch})")

# 3. Export weights and generate Fortran
exported = hs.extract_weights(result.model, in_norm, out_norm)
exported.save("mlp_surrogate.npz")

emitter = hs.FortranEmitter(exported)
emitter.write("nn_surrogate.f90")
```

The generated `nn_surrogate.f90` contains a self-contained Fortran 90 module with all weights baked in as `PARAMETER` arrays and `MATMUL`-based forward pass -- no external dependencies.

## Train an energy-based ICNN surrogate

Input-Convex Neural Networks (ICNN) guarantee convexity of the predicted strain energy, ensuring thermodynamic consistency.

```python
import hyper_surrogate as hs

material = hs.NeoHooke({"C10": 0.5, "KBULK": 1000.0})

# Energy target: the dataset stores (energy, dW/d_invariants) as a tuple
train_ds, val_ds, in_norm, out_norm = hs.create_datasets(
    material,
    n_samples=5000,
    input_type="invariants",
    target_type="energy",
)

# ICNN outputs a scalar (energy) -- stress is enforced via autograd in the loss
model = hs.ICNN(input_dim=3, hidden_dims=[32, 32])
result = hs.Trainer(
    model, train_ds, val_ds,
    loss_fn=hs.EnergyStressLoss(alpha=1.0, beta=1.0),
    max_epochs=500,
).fit()

# Export
exported = hs.extract_weights(result.model, in_norm, out_norm)
exported.save("icnn_surrogate.npz")
```

## Export a hybrid UMAT (NN energy + analytical mechanics)

The `HybridUMATEmitter` generates a complete Abaqus UMAT subroutine where the neural network provides the strain energy function `W(I1_bar, I2_bar, J)` and everything else — kinematics, PK2 stress, Cauchy push-forward, and spatial tangent with Jaumann correction — is computed analytically in Fortran.

```python
import numpy as np
import hyper_surrogate as hs
from hyper_surrogate.data.dataset import MaterialDataset, Normalizer
from hyper_surrogate.data.deformation import DeformationGenerator
from hyper_surrogate.mechanics.kinematics import Kinematics

# Generate training data with volumetric perturbation
material = hs.NeoHooke({"C10": 0.5, "KBULK": 1000.0})
gen = DeformationGenerator(seed=42)
F = gen.combined(10000, stretch_range=(0.8, 1.3), shear_range=(-0.2, 0.2))
rng = np.random.default_rng(99)
J_target = rng.uniform(0.95, 1.05, size=10000)
F = F * (J_target / np.linalg.det(F))[:, None, None] ** (1.0 / 3.0)
C = Kinematics.right_cauchy_green(F)

# Prepare invariant inputs and energy targets
i1 = Kinematics.isochoric_invariant1(C)
i2 = Kinematics.isochoric_invariant2(C)
j = np.sqrt(Kinematics.det_invariant(C))
inputs = np.column_stack([i1, i2, j])
energy = material.evaluate_energy(C)
dW_dI = material.evaluate_energy_grad_invariants(C)

in_norm = Normalizer().fit(inputs)
X = in_norm.transform(inputs).astype(np.float32)
W = energy.reshape(-1, 1).astype(np.float32)
S = (dW_dI * in_norm.params["std"]).astype(np.float32)

n_val = int(10000 * 0.15)
idx = np.random.default_rng(42).permutation(10000)
train_ds = MaterialDataset(X[idx[n_val:]], (W[idx[n_val:]], S[idx[n_val:]]))
val_ds = MaterialDataset(X[idx[:n_val]], (W[idx[:n_val]], S[idx[:n_val]]))

# Train energy MLP
model = hs.MLP(input_dim=3, output_dim=1, hidden_dims=[64, 64, 64], activation="softplus")
result = hs.Trainer(
    model, train_ds, val_ds,
    loss_fn=hs.EnergyStressLoss(alpha=1.0, beta=1.0),
    max_epochs=2000, lr=1e-3, patience=200, batch_size=512,
).fit()

# Export hybrid UMAT
energy_norm = Normalizer().fit(energy.reshape(-1, 1))
exported = hs.extract_weights(result.model, in_norm, energy_norm)
emitter = hs.HybridUMATEmitter(exported)
emitter.write("hybrid_umat.f90")
```

The generated `hybrid_umat.f90` contains a complete Fortran module with the NN forward/backward pass and a standard UMAT subroutine ready for Abaqus.

The generated `hybrid_umat.f90` contains a complete Fortran module with the NN forward/backward pass and a standard UMAT subroutine ready for Abaqus.

## Anisotropic model: Holzapfel-Ogden with fiber invariants

For transversely isotropic materials, the NN takes 5 invariants: `W(I1_bar, I2_bar, J, I4, I5)` where I4 and I5 are fiber invariants.

```python
import numpy as np
import hyper_surrogate as hs
from hyper_surrogate.mechanics.kinematics import Kinematics

# Define anisotropic material with fiber direction
fiber_dir = np.array([1.0, 0.0, 0.0])
material = hs.HolzapfelOgden(
    parameters={"a": 0.059, "b": 8.023, "af": 18.472, "bf": 16.026, "KBULK": 1000.0},
    fiber_direction=fiber_dir,
)

# Generate data and compute 5 invariants
gen = hs.DeformationGenerator(seed=42)
F = gen.combined(10000, stretch_range=(0.8, 1.3))
C = Kinematics.right_cauchy_green(F)

i4 = Kinematics.fiber_invariant4(C, fiber_dir)  # (N,)
i5 = Kinematics.fiber_invariant5(C, fiber_dir)  # (N,)

# Energy gradients: (N, 5) for anisotropic
dW_dI = material.evaluate_energy_grad_invariants(C)

# Train and export (same pipeline, input_dim=5)
# The HybridUMATEmitter auto-detects in_dim=5 and generates
# fiber invariant computation + dI4/dC, dI5/dC derivatives.
# Fiber direction is passed via props(1:3) in the Abaqus input file.
```

See `examples/train_holzapfel_ogden.py` for the complete runnable script.

See also the runnable scripts in the [`examples/`](https://github.com/jpsferreira/hyper-surrogate/tree/main/examples) directory.

## Kinematics utilities

Batch-compute common continuum mechanics quantities:

```python
import numpy as np
import hyper_surrogate as hs

gen = hs.DeformationGenerator(seed=0)
F = gen.combined(n=100)  # Mix of uniaxial, biaxial, shear modes

C = hs.Kinematics.right_cauchy_green(F)      # (100, 3, 3)
B = hs.Kinematics.left_cauchy_green(F)       # (100, 3, 3)
J = hs.Kinematics.jacobian(F)               # (100,)
I1 = hs.Kinematics.isochoric_invariant1(C)  # (100,)
I2 = hs.Kinematics.isochoric_invariant2(C)  # (100,)

stretches = hs.Kinematics.principal_stretches(C)  # (100, 3)
```

## Reporting

Visualize deformation data:

```python
from hyper_surrogate.reporting.reporter import Reporter

reporter = Reporter(C)  # (N, 3, 3) right Cauchy-Green batch

# Individual figures
reporter.fig_eigenvalues()          # eigenvalues of C (per-component)
reporter.fig_determinants()         # det(C) histogram
reporter.fig_invariants()           # I1_bar, I2_bar, J histograms
reporter.fig_principal_stretches()  # sorted principal stretches
reporter.fig_volume_change()        # J histogram with J=1 reference line

# Summary statistics
stats = reporter.basic_statistics()  # per-quantity mean, std, min, max

# Combined PDF report (all figures)
reporter.generate_report("report/")
```

You can also pass deformation gradients directly:

```python
reporter = Reporter(F, tensor_type="F")  # computes C = F^T F internally
reporter.generate_report("report/")
```
