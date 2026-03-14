"""Train an MLP to predict PK2 stress directly using StressLoss.

Supervised approach:
1. Generate deformation data and compute PK2 stress in Voigt notation
2. Train MLP mapping invariants -> PK2 stress (6 components)
3. Export to standalone Fortran 90 module

Usage:
    uv run python examples/train_neohooke_stress.py
"""

import hyper_surrogate as hs

# ── 1. Material and training data ─────────────────────────────────
print("── 1. Generating training data ──")
material = hs.NeoHooke({"C10": 0.5, "KBULK": 1000.0})

train_ds, val_ds, in_norm, out_norm = hs.create_datasets(
    material,
    n_samples=5000,
    input_type="invariants",  # 3 inputs: I1_bar, I2_bar, J
    target_type="pk2_voigt",  # 6 outputs: PK2 stress in Voigt notation
)

print(f"  Train samples: {len(train_ds)}")
print(f"  Val samples:   {len(val_ds)}")

# ── 2. Build and train the model ──────────────────────────────────
print("\n── 2. Training MLP ──")
model = hs.MLP(input_dim=3, output_dim=6, hidden_dims=[32, 32], activation="tanh")
result = hs.Trainer(
    model,
    train_ds,
    val_ds,
    loss_fn=hs.StressLoss(),
    max_epochs=500,
    lr=1e-3,
).fit()

best_val = result.history["val_loss"][result.best_epoch]
print(f"  Best val loss: {best_val:.6f} (epoch {result.best_epoch})")
print(f"  Total epochs:  {len(result.history["train_loss"])}")

# ── 3. Export to Fortran ──────────────────────────────────────────
print("\n── 3. Exporting ──")
exported = hs.extract_weights(result.model, in_norm, out_norm)
exported.save("mlp_neohooke_stress.npz")

emitter = hs.FortranEmitter(exported)
emitter.write("nn_neohooke_stress.f90")

print("  Weights:  mlp_neohooke_stress.npz")
print("  Fortran:  nn_neohooke_stress.f90")
