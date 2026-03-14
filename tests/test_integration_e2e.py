"""End-to-end integration tests.

Train a model, export weights, generate Fortran, and verify the NN forward
pass matches PyTorch numerically by reimplementing in numpy.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from hyper_surrogate.data.dataset import MaterialDataset, Normalizer  # noqa: E402
from hyper_surrogate.export.fortran.emitter import FortranEmitter  # noqa: E402
from hyper_surrogate.export.fortran.hybrid import HybridUMATEmitter  # noqa: E402
from hyper_surrogate.export.weights import ExportedModel, extract_weights  # noqa: E402
from hyper_surrogate.mechanics.kinematics import Kinematics  # noqa: E402
from hyper_surrogate.mechanics.materials import NeoHooke  # noqa: E402
from hyper_surrogate.models.icnn import ICNN  # noqa: E402
from hyper_surrogate.models.mlp import MLP  # noqa: E402
from hyper_surrogate.training.losses import EnergyStressLoss, StressLoss  # noqa: E402
from hyper_surrogate.training.trainer import Trainer  # noqa: E402

# ── Helpers ───────────────────────────────────────────────────────


def _numpy_forward_mlp(x, exported, activation="softplus"):
    """Reimplement MLP forward pass in numpy to validate against PyTorch."""
    act_fn = {
        "softplus": lambda a: np.log1p(np.exp(a)),
        "tanh": np.tanh,
        "relu": lambda a: np.maximum(0, a),
        "sigmoid": lambda a: 1.0 / (1.0 + np.exp(-a)),
    }[activation]

    # Normalize input
    if exported.input_normalizer:
        x = (x - exported.input_normalizer["mean"]) / exported.input_normalizer["std"]

    z = x
    for _i, layer in enumerate(exported.layers):
        W = exported.weights[layer.weights]
        b = exported.weights[layer.bias] if layer.bias else np.zeros(W.shape[0])
        z = z @ W.T + b
        if layer.activation != "identity":
            z = act_fn(z)

    # Denormalize output
    if exported.output_normalizer:
        z = z * exported.output_normalizer["std"] + exported.output_normalizer["mean"]
    return z


def _numpy_forward_icnn(x, exported, activation="softplus"):
    """Reimplement ICNN forward pass in numpy."""
    act_fn = {
        "softplus": lambda a: np.log1p(np.exp(a)),
        "relu": lambda a: np.maximum(0, a),
        "tanh": np.tanh,
    }[activation]

    x_input = x.copy()
    if exported.input_normalizer:
        x_input = (x_input - exported.input_normalizer["mean"]) / exported.input_normalizer["std"]

    weights = exported.weights
    layers = exported.layers

    # First layer: z = act(wx_0 @ x + bx_0)
    wx0 = weights[layers[0].weights]
    bx0 = weights[layers[0].bias]
    z = act_fn(x_input @ wx0.T + bx0)

    # Hidden layers: z = act(softplus(wz) @ z + wx @ x + bx)
    for layer in layers[1:-1]:
        wz_raw = weights[layer.weights]
        bx = weights[layer.bias]
        # Find corresponding wx layer
        # wx key: replace wz_layers with wx_layers, increment index by 1
        layer_idx = int(layer.weights.split(".")[1])
        wx_key = f"wx_layers.{layer_idx + 1}.weight"
        wx = weights[wx_key]
        wz = np.log1p(np.exp(wz_raw))  # softplus clamping
        z = act_fn(z @ wz.T + x_input @ wx.T + bx)

    # Output layer: softplus(wz_final) @ z + wx_final @ x + bx_final
    wz_final = weights[layers[-1].weights]
    bx_final = weights[layers[-1].bias]
    wx_final_key = "wx_final.weight"
    wx_final = weights[wx_final_key]
    wz_clamped = np.log1p(np.exp(wz_final))
    z = z @ wz_clamped.T + x_input @ wx_final.T + bx_final

    if exported.output_normalizer:
        z = z * exported.output_normalizer["std"] + exported.output_normalizer["mean"]
    return z


def _generate_training_data(n=500, seed=42):
    """Generate NeoHooke training data for integration tests."""
    material = NeoHooke({"C10": 0.5, "KBULK": 1000.0})
    from hyper_surrogate.data.deformation import DeformationGenerator

    gen = DeformationGenerator(seed=seed)
    F = gen.combined(n, stretch_range=(0.85, 1.2), shear_range=(-0.15, 0.15))
    rng = np.random.default_rng(99)
    J_target = rng.uniform(0.97, 1.03, size=n)
    J_current = np.linalg.det(F)
    F = F * (J_target / J_current)[:, None, None] ** (1.0 / 3.0)
    C = Kinematics.right_cauchy_green(F)

    i1 = Kinematics.isochoric_invariant1(C)
    i2 = Kinematics.isochoric_invariant2(C)
    j = np.sqrt(Kinematics.det_invariant(C))
    inputs = np.column_stack([i1, i2, j])

    energy = material.evaluate_energy(C)
    dW_dI = material.evaluate_energy_grad_invariants(C)

    return inputs, energy, dW_dI


# ── MLP end-to-end ────────────────────────────────────────────────


def test_mlp_train_export_numpy_match():
    """Train MLP, export, verify numpy forward pass matches PyTorch."""
    inputs, energy, dW_dI = _generate_training_data(n=300)

    in_norm = Normalizer().fit(inputs)
    X = in_norm.transform(inputs).astype(np.float32)
    W = energy.reshape(-1, 1).astype(np.float32)
    S = (dW_dI * in_norm.params["std"]).astype(np.float32)

    n_val = 50
    idx = np.random.default_rng(42).permutation(len(X))
    train_ds = MaterialDataset(X[idx[n_val:]], (W[idx[n_val:]], S[idx[n_val:]]))
    val_ds = MaterialDataset(X[idx[:n_val]], (W[idx[:n_val]], S[idx[:n_val]]))

    model = MLP(input_dim=3, output_dim=1, hidden_dims=[16, 16], activation="softplus")
    result = Trainer(
        model,
        train_ds,
        val_ds,
        loss_fn=EnergyStressLoss(alpha=1.0, beta=1.0),
        max_epochs=50,
        lr=1e-3,
        batch_size=128,
    ).fit()

    energy_norm = Normalizer().fit(energy.reshape(-1, 1))
    exported = extract_weights(result.model, in_norm, energy_norm)

    # Test on a few points
    test_inputs = inputs[idx[:10]]
    model.eval()
    x_torch = torch.tensor(in_norm.transform(test_inputs).astype(np.float32))
    with torch.no_grad():
        y_torch = model(x_torch).numpy()
    y_torch_denorm = y_torch * energy_norm.params["std"] + energy_norm.params["mean"]

    y_numpy = _numpy_forward_mlp(test_inputs, exported)

    np.testing.assert_allclose(y_numpy, y_torch_denorm, rtol=1e-5)


def test_mlp_stress_train_export():
    """Train MLP on PK2 stress, verify export and numpy match."""
    inputs, _, dW_dI = _generate_training_data(n=200)

    in_norm = Normalizer().fit(inputs)
    out_norm = Normalizer().fit(dW_dI)
    X = in_norm.transform(inputs).astype(np.float32)
    Y = out_norm.transform(dW_dI).astype(np.float32)

    n_val = 30
    idx = np.random.default_rng(42).permutation(len(X))
    train_ds = MaterialDataset(X[idx[n_val:]], Y[idx[n_val:]])
    val_ds = MaterialDataset(X[idx[:n_val]], Y[idx[:n_val]])

    model = MLP(input_dim=3, output_dim=3, hidden_dims=[16, 16], activation="tanh")
    result = Trainer(
        model,
        train_ds,
        val_ds,
        loss_fn=StressLoss(),
        max_epochs=30,
        lr=1e-3,
    ).fit()

    exported = extract_weights(result.model, in_norm, out_norm)
    test_inputs = inputs[idx[:5]]

    model.eval()
    x_torch = torch.tensor(in_norm.transform(test_inputs).astype(np.float32))
    with torch.no_grad():
        y_torch = model(x_torch).numpy()
    y_torch_denorm = y_torch * out_norm.params["std"] + out_norm.params["mean"]

    y_numpy = _numpy_forward_mlp(test_inputs, exported, activation="tanh")
    np.testing.assert_allclose(y_numpy, y_torch_denorm, rtol=1e-4)


# ── ICNN end-to-end ───────────────────────────────────────────────


def test_icnn_train_export_numpy_match():
    """Train ICNN, export, verify numpy forward pass matches PyTorch."""
    inputs, energy, dW_dI = _generate_training_data(n=300)

    in_norm = Normalizer().fit(inputs)
    X = in_norm.transform(inputs).astype(np.float32)
    W = energy.reshape(-1, 1).astype(np.float32)
    S = (dW_dI * in_norm.params["std"]).astype(np.float32)

    n_val = 50
    idx = np.random.default_rng(42).permutation(len(X))
    train_ds = MaterialDataset(X[idx[n_val:]], (W[idx[n_val:]], S[idx[n_val:]]))
    val_ds = MaterialDataset(X[idx[:n_val]], (W[idx[:n_val]], S[idx[:n_val]]))

    model = ICNN(input_dim=3, hidden_dims=[16, 16], activation="softplus")
    result = Trainer(
        model,
        train_ds,
        val_ds,
        loss_fn=EnergyStressLoss(alpha=1.0, beta=1.0),
        max_epochs=50,
        lr=1e-3,
        batch_size=128,
    ).fit()

    energy_norm = Normalizer().fit(energy.reshape(-1, 1))
    exported = extract_weights(result.model, in_norm, energy_norm)

    test_inputs = inputs[idx[:10]]
    model.eval()
    x_torch = torch.tensor(in_norm.transform(test_inputs).astype(np.float32))
    with torch.no_grad():
        y_torch = model(x_torch).numpy()
    y_torch_denorm = y_torch * energy_norm.params["std"] + energy_norm.params["mean"]

    y_numpy = _numpy_forward_icnn(test_inputs, exported)
    np.testing.assert_allclose(y_numpy, y_torch_denorm, rtol=1e-5)


# ── Export + Fortran generation ───────────────────────────────────


def test_mlp_export_generates_valid_fortran():
    """Full pipeline: train -> export -> FortranEmitter produces valid code."""
    inputs, energy, dW_dI = _generate_training_data(n=100)

    in_norm = Normalizer().fit(inputs)
    X = in_norm.transform(inputs).astype(np.float32)
    W = energy.reshape(-1, 1).astype(np.float32)
    S = (dW_dI * in_norm.params["std"]).astype(np.float32)

    train_ds = MaterialDataset(X[:80], (W[:80], S[:80]))
    val_ds = MaterialDataset(X[80:], (W[80:], S[80:]))

    model = MLP(input_dim=3, output_dim=1, hidden_dims=[8, 8], activation="softplus")
    result = Trainer(
        model,
        train_ds,
        val_ds,
        loss_fn=EnergyStressLoss(alpha=1.0, beta=1.0),
        max_epochs=5,
        lr=1e-3,
    ).fit()

    energy_norm = Normalizer().fit(energy.reshape(-1, 1))
    exported = extract_weights(result.model, in_norm, energy_norm)
    code = FortranEmitter(exported).emit()

    assert "MODULE nn_surrogate" in code
    assert "SUBROUTINE nn_forward" in code
    assert "w0" in code
    assert "in_mean" in code


def test_mlp_export_generates_valid_hybrid_umat():
    """Full pipeline: train -> export -> HybridUMATEmitter produces valid UMAT."""
    inputs, energy, dW_dI = _generate_training_data(n=100)

    in_norm = Normalizer().fit(inputs)
    X = in_norm.transform(inputs).astype(np.float32)
    W = energy.reshape(-1, 1).astype(np.float32)
    S = (dW_dI * in_norm.params["std"]).astype(np.float32)

    train_ds = MaterialDataset(X[:80], (W[:80], S[:80]))
    val_ds = MaterialDataset(X[80:], (W[80:], S[80:]))

    model = MLP(input_dim=3, output_dim=1, hidden_dims=[8, 8], activation="softplus")
    result = Trainer(
        model,
        train_ds,
        val_ds,
        loss_fn=EnergyStressLoss(alpha=1.0, beta=1.0),
        max_epochs=5,
        lr=1e-3,
    ).fit()

    energy_norm = Normalizer().fit(energy.reshape(-1, 1))
    exported = extract_weights(result.model, in_norm, energy_norm)
    code = HybridUMATEmitter(exported).emit()

    assert "MODULE nn_sef" in code
    assert "SUBROUTINE nn_eval" in code
    assert "SUBROUTINE umat" in code
    assert "d2W_dI2" in code
    assert "Cauchy" in code


def test_hybrid_umat_write_file(tmp_path):
    """Full pipeline including file write."""
    inputs, energy, dW_dI = _generate_training_data(n=100)

    in_norm = Normalizer().fit(inputs)
    energy_norm = Normalizer().fit(energy.reshape(-1, 1))
    X = in_norm.transform(inputs).astype(np.float32)
    W = energy.reshape(-1, 1).astype(np.float32)
    S = (dW_dI * in_norm.params["std"]).astype(np.float32)

    train_ds = MaterialDataset(X[:80], (W[:80], S[:80]))
    val_ds = MaterialDataset(X[80:], (W[80:], S[80:]))

    model = MLP(input_dim=3, output_dim=1, hidden_dims=[8], activation="softplus")
    result = Trainer(
        model,
        train_ds,
        val_ds,
        loss_fn=EnergyStressLoss(alpha=1.0, beta=1.0),
        max_epochs=3,
        lr=1e-3,
    ).fit()

    exported = extract_weights(result.model, in_norm, energy_norm)

    # Write both files
    f90_path = tmp_path / "nn_surrogate.f90"
    umat_path = tmp_path / "hybrid_umat.f90"
    npz_path = tmp_path / "weights.npz"

    FortranEmitter(exported).write(str(f90_path))
    HybridUMATEmitter(exported).write(str(umat_path))
    exported.save(str(npz_path))

    assert f90_path.exists()
    assert umat_path.exists()
    assert npz_path.exists()

    # Verify roundtrip
    loaded = ExportedModel.load(str(npz_path))
    assert loaded.metadata["architecture"] == "mlp"
    for k in exported.weights:
        np.testing.assert_array_equal(exported.weights[k], loaded.weights[k])


# ── Save/load preserves predictions ──────────────────────────────


def test_save_load_predictions_match():
    """Export, save, load, and verify numpy predictions are identical."""
    inputs, energy, _ = _generate_training_data(n=50)

    in_norm = Normalizer().fit(inputs)
    energy_norm = Normalizer().fit(energy.reshape(-1, 1))

    model = MLP(input_dim=3, output_dim=1, hidden_dims=[8, 8], activation="softplus")
    exported = extract_weights(model, in_norm, energy_norm)

    y_before = _numpy_forward_mlp(inputs[:5], exported)

    import tempfile

    with tempfile.TemporaryDirectory() as td:
        path = f"{td}/model.npz"
        exported.save(path)
        loaded = ExportedModel.load(path)

    y_after = _numpy_forward_mlp(inputs[:5], loaded)
    np.testing.assert_array_equal(y_before, y_after)
