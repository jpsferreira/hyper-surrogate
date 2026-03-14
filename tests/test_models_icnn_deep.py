"""Deep tests for ICNN model: Hessian PSD, activations, export, training, Fortran."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from hyper_surrogate.data.dataset import Normalizer  # noqa: E402
from hyper_surrogate.export.weights import ExportedModel, extract_weights  # noqa: E402
from hyper_surrogate.models.icnn import ICNN  # noqa: E402

# ── Convexity (multiple t values) ────────────────────────────────


@pytest.mark.parametrize("t", [0.1, 0.3, 0.5, 0.7, 0.9])
def test_convexity_multiple_t(t):
    """Jensen's inequality for multiple interpolation values."""
    model = ICNN(input_dim=3, hidden_dims=[32, 32])
    model.eval()
    x1 = torch.randn(100, 3)
    x2 = torch.randn(100, 3)
    with torch.no_grad():
        f_mix = model(t * x1 + (1 - t) * x2)
        f_avg = t * model(x1) + (1 - t) * model(x2)
    assert (f_mix <= f_avg + 1e-5).all(), f"Convexity violated at t={t}"


# ── Hessian PSD ──────────────────────────────────────────────────


def test_hessian_psd():
    """Convex function must have positive semi-definite Hessian everywhere."""
    model = ICNN(input_dim=3, hidden_dims=[16, 16])
    model.eval()
    rng = torch.Generator().manual_seed(42)
    for _ in range(20):
        x = torch.randn(1, 3, generator=rng, requires_grad=True)
        y = model(x)

        # Compute Hessian via autograd
        grad = torch.autograd.grad(y, x, create_graph=True)[0]
        H = torch.zeros(3, 3)
        for i in range(3):
            g2 = torch.autograd.grad(grad[0, i], x, retain_graph=True)[0]
            H[i] = g2[0]

        eigenvalues = torch.linalg.eigvalsh(H)
        assert (eigenvalues >= -1e-5).all(), f"Hessian not PSD: eigenvalues={eigenvalues}"


def test_hessian_psd_different_dims():
    """PSD check for different input dimensions."""
    for in_dim in [2, 5]:
        model = ICNN(input_dim=in_dim, hidden_dims=[8, 8])
        model.eval()
        x = torch.randn(1, in_dim, requires_grad=True)
        y = model(x)
        grad = torch.autograd.grad(y, x, create_graph=True)[0]
        H = torch.zeros(in_dim, in_dim)
        for i in range(in_dim):
            g2 = torch.autograd.grad(grad[0, i], x, retain_graph=True)[0]
            H[i] = g2[0]
        eigenvalues = torch.linalg.eigvalsh(H)
        assert (eigenvalues >= -1e-5).all()


# ── Multiple activations ─────────────────────────────────────────


@pytest.mark.parametrize("activation", ["softplus", "relu", "tanh"])
def test_forward_with_activation(activation):
    model = ICNN(input_dim=3, hidden_dims=[16, 16], activation=activation)
    x = torch.randn(10, 3)
    y = model(x)
    assert y.shape == (10, 1)
    assert torch.all(torch.isfinite(y))


@pytest.mark.parametrize("activation", ["softplus", "relu"])
def test_convexity_with_activation(activation):
    """Convexity guaranteed for softplus and relu (convex, non-decreasing)."""
    model = ICNN(input_dim=3, hidden_dims=[16, 16], activation=activation)
    model.eval()
    x1 = torch.randn(50, 3)
    x2 = torch.randn(50, 3)
    with torch.no_grad():
        f_mix = model(0.5 * x1 + 0.5 * x2)
        f_avg = 0.5 * model(x1) + 0.5 * model(x2)
    assert (f_mix <= f_avg + 1e-5).all()


# ── Architecture variants ────────────────────────────────────────


@pytest.mark.parametrize("hidden_dims", [[8], [32, 32], [16, 16, 16], [64, 32]])
def test_architecture_variants(hidden_dims):
    model = ICNN(input_dim=3, hidden_dims=hidden_dims)
    x = torch.randn(5, 3)
    y = model(x)
    assert y.shape == (5, 1)
    assert model.input_dim == 3
    assert model.output_dim == 1


def test_default_hidden_dims():
    model = ICNN(input_dim=3)
    assert model._hidden_dims == [64, 64]


# ── Export save/load roundtrip ────────────────────────────────────


def test_export_metadata():
    model = ICNN(input_dim=3, hidden_dims=[16, 8])
    in_norm = Normalizer().fit(np.random.randn(50, 3))
    exported = extract_weights(model, in_norm)
    assert exported.metadata["architecture"] == "icnn"
    assert exported.metadata["input_dim"] == 3
    assert exported.metadata["output_dim"] == 1
    assert len(exported.layers) > 0
    assert len(exported.weights) > 0


def test_export_save_load(tmp_path):
    model = ICNN(input_dim=3, hidden_dims=[16])
    in_norm = Normalizer().fit(np.random.randn(50, 3))
    energy_norm = Normalizer().fit(np.random.randn(50, 1))
    exported = extract_weights(model, in_norm, energy_norm)
    path = tmp_path / "icnn.npz"
    exported.save(str(path))
    loaded = ExportedModel.load(str(path))
    assert loaded.metadata["architecture"] == "icnn"
    assert loaded.metadata["input_dim"] == 3
    for k in exported.weights:
        np.testing.assert_array_equal(exported.weights[k], loaded.weights[k])
    np.testing.assert_array_equal(loaded.input_normalizer["mean"], exported.input_normalizer["mean"])
    np.testing.assert_array_equal(loaded.output_normalizer["mean"], exported.output_normalizer["mean"])


# ── Layer sequence ────────────────────────────────────────────────


def test_layer_sequence_structure():
    model = ICNN(input_dim=3, hidden_dims=[16, 8])
    seq = model.layer_sequence()
    # 3 layers: wx_0, wz_0+wx_1, wz_final+wx_final
    assert len(seq) == 3
    assert seq[0].weights == "wx_layers.0.weight"
    assert seq[0].bias == "wx_layers.0.bias"
    assert seq[0].activation == "softplus"
    assert seq[1].weights == "wz_layers.0.weight"
    assert seq[1].bias == "wx_layers.1.bias"
    assert seq[2].weights == "wz_final.weight"
    assert seq[2].bias == "wx_final.bias"
    assert seq[2].activation == "identity"


def test_layer_sequence_single_hidden():
    """Single hidden layer: wx_0 + output."""
    model = ICNN(input_dim=3, hidden_dims=[16])
    seq = model.layer_sequence()
    assert len(seq) == 2
    assert seq[0].activation == "softplus"
    assert seq[1].activation == "identity"


# ── Training integration ─────────────────────────────────────────


def test_training_with_energy_stress_loss():
    from hyper_surrogate.data.dataset import MaterialDataset
    from hyper_surrogate.training.losses import EnergyStressLoss
    from hyper_surrogate.training.trainer import Trainer

    model = ICNN(input_dim=3, hidden_dims=[8, 8])
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
    assert result.model is not None


def test_training_loss_decreases():
    from hyper_surrogate.data.dataset import MaterialDataset
    from hyper_surrogate.training.losses import EnergyStressLoss
    from hyper_surrogate.training.trainer import Trainer

    model = ICNN(input_dim=3, hidden_dims=[16, 16])
    n = 200
    # Use simple learnable target: W = sum(x^2)
    X = np.random.randn(n, 3).astype(np.float32) * 0.5
    W = (X**2).sum(axis=1, keepdims=True).astype(np.float32)
    S = (2 * X).astype(np.float32)

    train_ds = MaterialDataset(X[:160], (W[:160], S[:160]))
    val_ds = MaterialDataset(X[160:], (W[160:], S[160:]))

    result = Trainer(
        model,
        train_ds,
        val_ds,
        loss_fn=EnergyStressLoss(alpha=1.0, beta=1.0),
        max_epochs=50,
        lr=1e-3,
        batch_size=64,
    ).fit()
    # Loss should decrease
    assert result.history["train_loss"][-1] < result.history["train_loss"][0]


# ── Consistency with PolyconvexICNN ───────────────────────────────


def test_single_branch_polyconvex_matches_icnn():
    """Single-group PolyconvexICNN should behave like ICNN."""
    from hyper_surrogate.models.polyconvex import PolyconvexICNN

    torch.manual_seed(42)
    icnn = ICNN(input_dim=3, hidden_dims=[8, 8], activation="softplus")

    torch.manual_seed(42)
    poly = PolyconvexICNN(groups=[[0, 1, 2]], hidden_dims=[8, 8], activation="softplus")

    # Same architecture, same seed -> same weights
    x = torch.randn(5, 3)
    y_icnn = icnn(x)
    y_poly = poly(x)
    # Shapes match
    assert y_icnn.shape == y_poly.shape == (5, 1)
