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

print(f"Best validation loss: {result.best_val_loss:.6f} (epoch {result.best_epoch})")

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

reporter = Reporter(C)  # (N, 3, 3) tensor batch
reporter.fig_eigenvalues()
reporter.fig_determinants()
reporter.generate_report("report/")
```
