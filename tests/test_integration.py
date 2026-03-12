import os
import tempfile

import pytest

torch = pytest.importorskip("torch")


def test_mlp_end_to_end():
    """Full pipeline: data -> train -> export -> Fortran."""
    import hyper_surrogate as hs

    material = hs.NeoHooke({"C10": 0.5, "KBULK": 1000.0})
    train_ds, val_ds, in_norm, out_norm = hs.create_datasets(
        material,
        n_samples=200,
        input_type="invariants",
        target_type="pk2_voigt",
    )

    model = hs.MLP(input_dim=3, output_dim=6, hidden_dims=[16], activation="tanh")
    result = hs.Trainer(model, train_ds, val_ds, loss_fn=hs.StressLoss(), max_epochs=10).fit()

    exported = hs.extract_weights(result.model, in_norm, out_norm)

    with tempfile.TemporaryDirectory() as td:
        # Save/load weights
        npz_path = os.path.join(td, "model.npz")
        exported.save(npz_path)
        loaded = hs.ExportedModel.load(npz_path)

        # Generate Fortran
        f90_path = os.path.join(td, "nn_surrogate.f90")
        hs.FortranEmitter(loaded).write(f90_path)
        assert os.path.exists(f90_path)
        with open(f90_path) as f:
            code = f.read()
        assert "MODULE nn_surrogate" in code
        assert "MATMUL" in code


def test_icnn_end_to_end():
    """ICNN pipeline: data -> train -> export."""
    import hyper_surrogate as hs

    material = hs.NeoHooke({"C10": 0.5, "KBULK": 1000.0})
    train_ds, val_ds, in_norm, out_norm = hs.create_datasets(
        material,
        n_samples=200,
        input_type="invariants",
        target_type="energy",
    )

    model = hs.ICNN(input_dim=3, hidden_dims=[16])
    result = hs.Trainer(model, train_ds, val_ds, loss_fn=hs.EnergyStressLoss(), max_epochs=10).fit()

    exported = hs.extract_weights(result.model, in_norm, out_norm)
    assert exported.metadata["architecture"] == "icnn"
