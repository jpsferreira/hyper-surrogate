"""Train an MLP on the Holzapfel-Ogden anisotropic strain energy function.

Demonstrates the anisotropic pipeline:
1. Generate deformations with fiber direction
2. Compute 5 invariants: I1_bar, I2_bar, J, I4, I5
3. Train MLP on W(I1_bar, I2_bar, J, I4, I5) with energy + stress loss
4. Export to anisotropic hybrid UMAT

Usage:
    uv run python examples/train_holzapfel_ogden.py
"""

import numpy as np
import torch
from sympy import Symbol, lambdify

import hyper_surrogate as hs
from hyper_surrogate.data.dataset import MaterialDataset, Normalizer
from hyper_surrogate.data.deformation import DeformationGenerator
from hyper_surrogate.mechanics.kinematics import Kinematics

# ── 1. Anisotropic material and deformation data ─────────────────
fiber_dir = np.array([1.0, 0.0, 0.0])
material = hs.HolzapfelOgden(
    parameters={"a": 0.059, "b": 8.023, "af": 18.472, "bf": 16.026, "KBULK": 1000.0},
    fiber_direction=fiber_dir,
)
n = 20000

gen = DeformationGenerator(seed=42)
F_base = gen.combined(n, stretch_range=(0.8, 1.3), shear_range=(-0.2, 0.2))

# Add volumetric perturbation
rng = np.random.default_rng(99)
J_target = rng.uniform(0.95, 1.05, size=n)
J_current = np.linalg.det(F_base)
F = F_base * (J_target / J_current)[:, None, None] ** (1.0 / 3.0)
C = Kinematics.right_cauchy_green(F)

# ── 2. Inputs: 5 invariants ──────────────────────────────────────
i1 = Kinematics.isochoric_invariant1(C)
i2 = Kinematics.isochoric_invariant2(C)
j = np.sqrt(Kinematics.det_invariant(C))
i4 = Kinematics.fiber_invariant4(C, fiber_dir)
i5 = Kinematics.fiber_invariant5(C, fiber_dir)
inputs = np.column_stack([i1, i2, j, i4, i5])

# Stress targets: dW/dI for 5 invariants
dW_dI = material.evaluate_energy_grad_invariants(C)  # (N, 5)

print(f"Samples: {n}")
print(f"I1_bar=[{i1.min():.3f}, {i1.max():.3f}]")
print(f"I2_bar=[{i2.min():.3f}, {i2.max():.3f}]")
print(f"J     =[{j.min():.4f}, {j.max():.4f}]")
print(f"I4    =[{i4.min():.4f}, {i4.max():.4f}]")
print(f"I5    =[{i5.min():.4f}, {i5.max():.4f}]")

# ── 3. Normalize and build datasets ──────────────────────────────
in_norm = Normalizer().fit(inputs)
X = in_norm.transform(inputs).astype(np.float32)

# For energy loss we need energy values — compute via sef_from_invariants
i1s, i2s, js, i4s, i5s = Symbol("I1b"), Symbol("I2b"), Symbol("J"), Symbol("I4"), Symbol("I5")
W_sym = material.sef_from_invariants(i1s, i2s, js, i4s, i5s)
param_syms = list(material._symbols.values())
W_func = lambdify((i1s, i2s, js, i4s, i5s, *param_syms), W_sym, modules="numpy")
param_vals = list(material._params.values())
energy = W_func(i1, i2, j, i4, i5, *param_vals).astype(np.float64)
W = energy.reshape(-1, 1).astype(np.float32)

# Scale stress targets to normalized-input space
S = (dW_dI * in_norm.params["std"]).astype(np.float32)

n_val = int(n * 0.15)
idx = np.random.default_rng(42).permutation(n)
ti, vi = idx[n_val:], idx[:n_val]

train_ds = MaterialDataset(X[ti], (W[ti], S[ti]))
val_ds = MaterialDataset(X[vi], (W[vi], S[vi]))

# ── 4. Train ─────────────────────────────────────────────────────
model = hs.MLP(input_dim=5, output_dim=1, hidden_dims=[64, 64, 64], activation="softplus")
result = hs.Trainer(
    model,
    train_ds,
    val_ds,
    loss_fn=hs.EnergyStressLoss(alpha=1.0, beta=1.0),
    max_epochs=3000,
    lr=1e-3,
    patience=300,
    batch_size=512,
).fit()

best_val = result.history["val_loss"][result.best_epoch]
print(f"\nBest val loss: {best_val:.6f} (epoch {result.best_epoch})")
print(f"Total epochs: {len(result.history["train_loss"])}")

# ── 5. Evaluate ──────────────────────────────────────────────────
print("\n── Hybrid inference: invariants -> MLP(W) -> autodiff -> dW/dI ──")
model.eval()
test_idx = vi[:500]

test_inp = inputs[test_idx]
x = torch.tensor(in_norm.transform(test_inp).astype(np.float32), requires_grad=True)

W_pred = model(x)
dW_dx = torch.autograd.grad(W_pred.sum(), x, create_graph=True)[0]

std_t = torch.tensor(in_norm.params["std"], dtype=torch.float32)
dW_dI_pred = (dW_dx / std_t).detach().numpy()
dW_dI_ref = dW_dI[test_idx]

names = ["dW/dI1_bar", "dW/dI2_bar", "dW/dJ", "dW/dI4", "dW/dI5"]
for i, name in enumerate(names):
    ss_res = np.sum((dW_dI_pred[:, i] - dW_dI_ref[:, i]) ** 2)
    ss_tot = np.sum((dW_dI_ref[:, i] - dW_dI_ref[:, i].mean()) ** 2)
    r2 = 1 - ss_res / max(ss_tot, 1e-30)
    print(f"  {name}: R²={r2:.6f}")

# ── 6. Export: anisotropic hybrid UMAT ───────────────────────────
energy_norm = Normalizer().fit(energy.reshape(-1, 1))
exported = hs.extract_weights(result.model, in_norm, energy_norm)
exported.save("mlp_holzapfel_ogden.npz")

emitter = hs.HybridUMATEmitter(exported)
emitter.write("hybrid_umat_holzapfel_ogden.f90")

print("\nExported:")
print("  Weights:  mlp_holzapfel_ogden.npz")
print("  Fortran:  hybrid_umat_holzapfel_ogden.f90")
print(f"  Fiber:    a0 = {fiber_dir} (passed via props(1:3) in Abaqus)")
