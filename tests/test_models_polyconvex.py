"""Tests for PolyconvexICNN model."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from hyper_surrogate.data.dataset import Normalizer  # noqa: E402
from hyper_surrogate.export.fortran.emitter import FortranEmitter  # noqa: E402
from hyper_surrogate.export.fortran.hybrid import HybridUMATEmitter  # noqa: E402
from hyper_surrogate.export.weights import extract_weights  # noqa: E402
from hyper_surrogate.models.polyconvex import PolyconvexICNN  # noqa: E402


def test_forward_shape():
    model = PolyconvexICNN(groups=[[0], [1], [2]], hidden_dims=[8, 8])
    x = torch.randn(10, 3)
    y = model(x)
    assert y.shape == (10, 1)


def test_forward_shape_aniso():
    model = PolyconvexICNN(groups=[[0], [1], [2], [3, 4]], hidden_dims=[8, 8])
    x = torch.randn(10, 5)
    y = model(x)
    assert y.shape == (10, 1)


def test_input_output_dim():
    model = PolyconvexICNN(groups=[[0], [1], [2]])
    assert model.input_dim == 3
    assert model.output_dim == 1


def test_overlapping_groups_rejected():
    with pytest.raises(ValueError, match="overlap"):
        PolyconvexICNN(groups=[[0, 1], [1, 2]])


def test_groups_property():
    groups = [[0], [1], [2], [3, 4]]
    model = PolyconvexICNN(groups=groups)
    assert model.groups == groups


def test_per_branch_convexity():
    """Each branch should be convex (Jensen's inequality)."""
    model = PolyconvexICNN(groups=[[0], [1], [2]], hidden_dims=[16, 16])
    model.eval()

    x1 = torch.randn(50, 3)
    x2 = torch.randn(50, 3)
    lam = 0.3

    y_mid = model(lam * x1 + (1 - lam) * x2)
    y_conv = lam * model(x1) + (1 - lam) * model(x2)

    # Convex: f(lam*x1 + (1-lam)*x2) <= lam*f(x1) + (1-lam)*f(x2)
    assert (y_mid <= y_conv + 1e-5).all(), "Convexity violated"


def test_gradient_exists():
    model = PolyconvexICNN(groups=[[0], [1], [2]])
    x = torch.randn(5, 3, requires_grad=True)
    y = model(x)
    y.sum().backward()
    assert x.grad is not None
    assert x.grad.shape == (5, 3)


def test_branch_sequence():
    model = PolyconvexICNN(groups=[[0], [1], [2]])
    branches = model.branch_sequence()
    assert branches is not None
    assert len(branches) == 3
    assert branches[0].name == "branch_0"
    assert branches[0].input_indices == [0]
    assert branches[2].input_indices == [2]
    # Each branch should have layers with prefixed weight keys
    for b in branches:
        for layer in b.layers:
            assert layer.weights.startswith(f"branches.{b.name[-1]}.")


def test_layer_sequence():
    """layer_sequence returns first branch layers (backward compat)."""
    model = PolyconvexICNN(groups=[[0], [1], [2]])
    layers = model.layer_sequence()
    assert len(layers) > 0
    assert all(ly.weights.startswith("branches.0.") for ly in layers)


def test_export_weights():
    model = PolyconvexICNN(groups=[[0], [1], [2]], hidden_dims=[8])
    in_norm = Normalizer().fit(np.random.randn(50, 3))
    exported = extract_weights(model, in_norm)
    assert exported.metadata["architecture"] == "polyconvexicnn"
    assert exported.metadata["input_dim"] == 3
    assert exported.metadata["output_dim"] == 1
    assert "branches" in exported.metadata
    assert len(exported.metadata["branches"]) == 3


def test_export_save_load(tmp_path):
    model = PolyconvexICNN(groups=[[0], [1], [2]], hidden_dims=[8])
    in_norm = Normalizer().fit(np.random.randn(50, 3))
    exported = extract_weights(model, in_norm)
    path = tmp_path / "poly.npz"
    exported.save(str(path))
    loaded = type(exported).load(str(path))
    assert loaded.metadata["branches"] == exported.metadata["branches"]
    assert loaded.metadata["architecture"] == "polyconvexicnn"


def test_fortran_emitter_polyconvex():
    model = PolyconvexICNN(groups=[[0], [1], [2]], hidden_dims=[8, 8])
    in_norm = Normalizer().fit(np.random.randn(50, 3))
    exported = extract_weights(model, in_norm)
    code = FortranEmitter(exported).emit()
    assert "MODULE nn_surrogate" in code
    assert "Branch 0" in code
    assert "Branch 1" in code
    assert "Branch 2" in code
    assert "grad_b0" in code
    assert "stress" in code


def test_hybrid_umat_polyconvex():
    model = PolyconvexICNN(groups=[[0], [1], [2]], hidden_dims=[8, 8])
    in_norm = Normalizer().fit(np.random.randn(50, 3))
    energy_norm = Normalizer().fit(np.random.randn(50, 1))
    exported = extract_weights(model, in_norm, energy_norm)
    code = HybridUMATEmitter(exported).emit()
    assert "MODULE nn_sef" in code
    assert "SUBROUTINE nn_eval" in code
    assert "SUBROUTINE umat" in code
    # Per-branch forward/backward
    assert "Branch 0" in code
    assert "Hessian branch" in code
    # Block-diagonal scatter
    assert "d2W_dI2" in code


def test_hybrid_umat_polyconvex_aniso():
    """Anisotropic polyconvex UMAT (in_dim=5)."""
    model = PolyconvexICNN(groups=[[0], [1], [2], [3, 4]], hidden_dims=[8, 8])
    in_norm = Normalizer().fit(np.random.randn(50, 5))
    energy_norm = Normalizer().fit(np.random.randn(50, 1))
    exported = extract_weights(model, in_norm, energy_norm)
    code = HybridUMATEmitter(exported).emit()
    assert "nn_input(5)" in code
    assert "dW_dI(5)" in code
    assert "d2W_dI2(5,5)" in code
    assert "I4" in code
    assert "Branch 3" in code


def test_training_with_energy_stress_loss():
    """PolyconvexICNN can be trained with EnergyStressLoss."""
    from hyper_surrogate.data.dataset import MaterialDataset
    from hyper_surrogate.training.losses import EnergyStressLoss
    from hyper_surrogate.training.trainer import Trainer

    model = PolyconvexICNN(groups=[[0], [1], [2]], hidden_dims=[8, 8])
    n = 100
    X = np.random.randn(n, 3).astype(np.float32)
    W = np.random.randn(n, 1).astype(np.float32)
    S = np.random.randn(n, 3).astype(np.float32)

    train_ds = MaterialDataset(X[:80], (W[:80], S[:80]))
    val_ds = MaterialDataset(X[80:], (W[80:], S[80:]))

    result = Trainer(
        model,
        train_ds,
        val_ds,
        loss_fn=EnergyStressLoss(alpha=1.0, beta=1.0),
        max_epochs=5,
        lr=1e-3,
    ).fit()
    assert len(result.history["train_loss"]) == 5
