"""End-to-end: train an MLP on strain energy and export a hybrid UMAT.

The hybrid UMAT combines:
- NN-based strain energy function W(I1_bar, I2_bar, J)
- Analytical kinematics: F -> C -> invariants
- Analytical stress: PK2 = 2 * sum(dW/dIk * dIk/dC), Cauchy = (1/J) F S F^T
- Analytical tangent: material tangent + push-forward + Jaumann correction

Usage:
    uv run python examples/export_hybrid_umat.py
"""

import numpy as np

import hyper_surrogate as hs
from hyper_surrogate.data.dataset import MaterialDataset, Normalizer
from hyper_surrogate.data.deformation import DeformationGenerator
from hyper_surrogate.mechanics.kinematics import Kinematics

# ── 1. Material and deformation data ──────────────────────────────
print("── 1. Generating training data ──")
material = hs.NeoHooke({"C10": 0.5, "KBULK": 1000.0})
n = 10000

gen = DeformationGenerator(seed=42)
F_base = gen.combined(n, stretch_range=(0.8, 1.3), shear_range=(-0.2, 0.2))

# Add volumetric perturbation
rng = np.random.default_rng(99)
J_target = rng.uniform(0.95, 1.05, size=n)
J_current = np.linalg.det(F_base)
F = F_base * (J_target / J_current)[:, None, None] ** (1.0 / 3.0)
C = Kinematics.right_cauchy_green(F)

# ── 2. Inputs and targets ─────────────────────────────────────────
i1 = Kinematics.isochoric_invariant1(C)
i2 = Kinematics.isochoric_invariant2(C)
j = np.sqrt(Kinematics.det_invariant(C))
inputs = np.column_stack([i1, i2, j])

energy = material.evaluate_energy(C)
dW_dI = material.evaluate_energy_grad_invariants(C)

print(f"  Samples: {n}")
print(f"  I1=[{i1.min():.3f}, {i1.max():.3f}]")
print(f"  J =[{j.min():.4f}, {j.max():.4f}]")

# ── 3. Normalize and build datasets ───────────────────────────────
in_norm = Normalizer().fit(inputs)
X = in_norm.transform(inputs).astype(np.float32)
W = energy.reshape(-1, 1).astype(np.float32)
S = (dW_dI * in_norm.params["std"]).astype(np.float32)

n_val = int(n * 0.15)
idx = np.random.default_rng(42).permutation(n)
ti, vi = idx[n_val:], idx[:n_val]

train_ds = MaterialDataset(X[ti], (W[ti], S[ti]))
val_ds = MaterialDataset(X[vi], (W[vi], S[vi]))

# ── 4. Train ──────────────────────────────────────────────────────
print("\n── 2. Training MLP ──")
model = hs.MLP(input_dim=3, output_dim=1, hidden_dims=[64, 64, 64], activation="softplus")
result = hs.Trainer(
    model,
    train_ds,
    val_ds,
    loss_fn=hs.EnergyStressLoss(alpha=1.0, beta=1.0),
    max_epochs=2000,
    lr=1e-3,
    patience=200,
    batch_size=512,
).fit()

best_val = result.history["val_loss"][result.best_epoch]
print(f"  Best val loss: {best_val:.6f} (epoch {result.best_epoch})")

# ── 5. Export hybrid UMAT ─────────────────────────────────────────
print("\n── 3. Exporting hybrid UMAT ──")
energy_norm = Normalizer().fit(energy.reshape(-1, 1))
exported = hs.extract_weights(result.model, in_norm, energy_norm)
exported.save("mlp_hybrid_sef.npz")

emitter = hs.HybridUMATEmitter(exported)
emitter.write("hybrid_umat.f90")

print("  Weights:  mlp_hybrid_sef.npz")
print("  UMAT:     hybrid_umat.f90")
print("\nThe generated UMAT contains:")
print("  - NN forward + backward pass (dW/dI via backprop)")
print("  - Analytical kinematics (F -> C -> invariants)")
print("  - Analytical PK2 stress and Cauchy push-forward")
print("  - Material tangent with Jaumann correction")
