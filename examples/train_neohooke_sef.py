"""Train an MLP to learn the NeoHooke strain energy function.

Hybrid approach:
1. Train MLP on W(I1_bar, I2_bar, J) using energy + stress gradient loss
2. At inference: F -> C -> invariants -> MLP(W) -> autodiff -> dW/dI
3. Export to Fortran for use in FE solvers

Usage:
    uv run python examples/train_neohooke_sef.py
"""

import numpy as np
import torch

import hyper_surrogate as hs
from hyper_surrogate.data.dataset import MaterialDataset, Normalizer
from hyper_surrogate.data.deformation import DeformationGenerator
from hyper_surrogate.mechanics.kinematics import Kinematics

# ── 1. Material and deformation data ───────────────────────────────
material = hs.NeoHooke({"C10": 0.5, "KBULK": 1000.0})
n = 20000

gen = DeformationGenerator(seed=42)
F_base = gen.combined(n, stretch_range=(0.8, 1.3), shear_range=(-0.2, 0.2))

# Add volumetric perturbation (combined generator is incompressible)
rng = np.random.default_rng(99)
J_target = rng.uniform(0.95, 1.05, size=n)
J_current = np.linalg.det(F_base)
F = F_base * (J_target / J_current)[:, None, None] ** (1.0 / 3.0)
C = Kinematics.right_cauchy_green(F)

# ── 2. Inputs and targets ──────────────────────────────────────────
i1 = Kinematics.isochoric_invariant1(C)
i2 = Kinematics.isochoric_invariant2(C)
j = np.sqrt(Kinematics.det_invariant(C))
inputs = np.column_stack([i1, i2, j])

energy = material.evaluate_energy(C)  # (N,)
dW_dI = material.evaluate_energy_grad_invariants(C)  # (N, 3)

print(f"Samples: {n}")
print(f"I1=[{i1.min():.3f}, {i1.max():.3f}]")
print(f"I2=[{i2.min():.3f}, {i2.max():.3f}]")
print(f"J =[{j.min():.4f}, {j.max():.4f}]")
print(f"Energy=[{energy.min():.4f}, {energy.max():.4f}]")

# ── 3. Normalize and build datasets ────────────────────────────────
in_norm = Normalizer().fit(inputs)
X = in_norm.transform(inputs).astype(np.float32)
W = energy.reshape(-1, 1).astype(np.float32)

# Scale stress targets to normalized-input space:
# dW/d(x_norm) = dW/dI * std_I (chain rule for standardization)
S = (dW_dI * in_norm.params["std"]).astype(np.float32)

n_val = int(n * 0.15)
idx = np.random.default_rng(42).permutation(n)
ti, vi = idx[n_val:], idx[:n_val]

train_ds = MaterialDataset(X[ti], (W[ti], S[ti]))
val_ds = MaterialDataset(X[vi], (W[vi], S[vi]))

# ── 4. Train ───────────────────────────────────────────────────────
model = hs.MLP(input_dim=3, output_dim=1, hidden_dims=[64, 64, 64], activation="softplus")
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

# ── 5. Evaluate: hybrid inference F -> stress ──────────────────────
print("\n── Hybrid inference: F -> invariants -> MLP(W) -> autodiff -> dW/dI ──")
model.eval()
test_idx = vi[:500]

test_inp = inputs[test_idx]
x = torch.tensor(in_norm.transform(test_inp).astype(np.float32), requires_grad=True)

W_pred = model(x)
dW_dx = torch.autograd.grad(W_pred.sum(), x, create_graph=True)[0]

# Convert back to raw invariant space: dW/dI = dW/d(x_norm) / std
std_t = torch.tensor(in_norm.params["std"], dtype=torch.float32)
dW_dI_pred = (dW_dx / std_t).detach().numpy()
dW_dI_ref = dW_dI[test_idx]

for i, name in enumerate(["dW/dI1_bar", "dW/dI2_bar", "dW/dJ"]):
    ss_res = np.sum((dW_dI_pred[:, i] - dW_dI_ref[:, i]) ** 2)
    ss_tot = np.sum((dW_dI_ref[:, i] - dW_dI_ref[:, i].mean()) ** 2)
    r2 = 1 - ss_res / max(ss_tot, 1e-30)
    mae = np.mean(np.abs(dW_dI_pred[:, i] - dW_dI_ref[:, i]))
    print(f"  {name}: R²={r2:.6f}, MAE={mae:.6f}")

W_p = W_pred.detach().numpy().flatten()
W_r = energy[test_idx]
r2_e = 1 - np.sum((W_p - W_r) ** 2) / np.sum((W_r - W_r.mean()) ** 2)
print(f"  Energy:    R²={r2_e:.6f}")

# ── 6. Export ──────────────────────────────────────────────────────
energy_norm = Normalizer().fit(energy.reshape(-1, 1))
exported = hs.extract_weights(result.model, in_norm, energy_norm)
exported.save("mlp_neohooke_sef.npz")

emitter = hs.FortranEmitter(exported)
emitter.write("nn_neohooke_sef.f90")

print("\nExported:")
print("  Weights:  mlp_neohooke_sef.npz")
print("  Fortran:  nn_neohooke_sef.f90")
