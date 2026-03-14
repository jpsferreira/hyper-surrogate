"""Train an ICNN to learn a convex strain energy function.

Energy-based approach with convexity guarantee:
1. Generate invariant inputs and energy + stress gradient targets
2. Train ICNN (Input-Convex Neural Network) with EnergyStressLoss
3. ICNN guarantees convexity of W(I) -> thermodynamic consistency
4. Export weights for downstream use

Usage:
    uv run python examples/train_icnn_energy.py
"""

import hyper_surrogate as hs

# ── 1. Material and training data ─────────────────────────────────
print("── 1. Generating training data ──")
material = hs.NeoHooke({"C10": 0.5, "KBULK": 1000.0})

# Energy target: dataset stores (energy, dW/d_invariants) as a tuple
train_ds, val_ds, in_norm, out_norm = hs.create_datasets(
    material,
    n_samples=5000,
    input_type="invariants",
    target_type="energy",
)

print(f"  Train samples: {len(train_ds)}")
print(f"  Val samples:   {len(val_ds)}")

# ── 2. Build and train the ICNN ───────────────────────────────────
print("\n── 2. Training ICNN ──")
# ICNN outputs a scalar (energy); stress is enforced via autograd in the loss
model = hs.ICNN(input_dim=3, hidden_dims=[32, 32])
result = hs.Trainer(
    model,
    train_ds,
    val_ds,
    loss_fn=hs.EnergyStressLoss(alpha=1.0, beta=1.0),
    max_epochs=500,
    lr=1e-3,
).fit()

best_val = result.history["val_loss"][result.best_epoch]
print(f"  Best val loss: {best_val:.6f} (epoch {result.best_epoch})")
print(f"  Total epochs:  {len(result.history["train_loss"])}")

# ── 3. Export weights ─────────────────────────────────────────────
print("\n── 3. Exporting ──")
exported = hs.extract_weights(result.model, in_norm, out_norm)
exported.save("icnn_neohooke_energy.npz")

print("  Weights: icnn_neohooke_energy.npz")
