"""CANN model discovery: identify minimal constitutive law from data.

Trains a CANN on NeoHooke synthetic data and uses sparsification
to discover which basis functions best describe the material behavior.
"""

from __future__ import annotations

from hyper_surrogate.data.dataset import create_datasets
from hyper_surrogate.mechanics.materials import NeoHooke
from hyper_surrogate.models.cann import CANN
from hyper_surrogate.training.losses import EnergyStressLoss
from hyper_surrogate.training.trainer import Trainer


def main() -> None:
    # Generate training data from NeoHooke
    material = NeoHooke(parameters={"C10": 0.5, "KBULK": 1000.0})
    train_ds, val_ds, in_norm, out_norm = create_datasets(
        material,
        n_samples=5000,
        input_type="invariants",
        target_type="energy",
        seed=42,
    )

    print(f"Training samples: {len(train_ds)}")
    print(f"Input dim: {train_ds.inputs.shape[1]}")

    # Create CANN with various basis functions
    model = CANN(
        input_dim=train_ds.inputs.shape[1],
        n_polynomial=3,
        n_exponential=2,
        use_logarithmic=True,
        learnable_exponents=True,
    )

    # Train with energy-stress loss
    loss_fn = EnergyStressLoss(alpha=1.0, beta=1.0)
    trainer = Trainer(model, train_ds, val_ds, loss_fn=loss_fn, max_epochs=500)
    result = trainer.fit()

    print("\nTraining complete:")
    train_loss = result.history["train_loss"][-1]
    val_loss = result.history["val_loss"][-1]
    print(f"  Final train loss: {train_loss:.6f}")
    print(f"  Final val loss: {val_loss:.6f}")

    # Discover active terms
    print("\nActive basis functions (weight > 0.01):")
    terms = model.get_active_terms(threshold=0.01)
    for t in terms:
        inv_name = f"I{t["invariant"] + 1}_bar" if t["invariant"] < 2 else "J"
        w = t["weight"]
        if t["type"] == "polynomial":
            print(f"  w={w:.4f}: ({inv_name})^{t["power"]}")
        elif t["type"] == "exponential":
            print(f"  w={w:.4f}: exp({t["stiffness"]:.3f} * {inv_name}^2) - 1")
        elif t["type"] == "logarithmic":
            print(f"  w={w:.4f}: log(1 + {inv_name}^2)")

    print(f"\nTotal active terms: {len(terms)} / {model._n_basis}")
    print("Expected: dominant polynomial terms on I1_bar (NeoHooke ~ C10*(I1_bar - 3))")


if __name__ == "__main__":
    main()
