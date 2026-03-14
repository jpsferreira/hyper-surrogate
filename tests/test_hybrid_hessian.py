"""Tests for analytical Hessian in HybridUMATEmitter.

Validates the exact d²W/dI² computation against PyTorch autograd
for various activations and architectures.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch.autograd.functional import hessian  # noqa: E402

from hyper_surrogate.data.dataset import Normalizer  # noqa: E402
from hyper_surrogate.export.fortran.hybrid import HybridUMATEmitter  # noqa: E402
from hyper_surrogate.export.weights import extract_weights  # noqa: E402
from hyper_surrogate.models.mlp import MLP  # noqa: E402


def _make_model(activation="softplus", hidden_dims=None):
    if hidden_dims is None:
        hidden_dims = [8, 8]
    model = MLP(input_dim=3, output_dim=1, hidden_dims=hidden_dims, activation=activation)
    in_norm = Normalizer().fit(np.random.default_rng(42).standard_normal((50, 3)))
    energy_norm = Normalizer().fit(np.random.default_rng(42).standard_normal((50, 1)))
    return model, in_norm, energy_norm


def _pytorch_hessian(model, in_norm, x_raw):
    """Compute d²W/dI² using PyTorch autograd on raw (un-normalized) inputs."""
    model.eval()
    std = torch.tensor(in_norm.params["std"], dtype=torch.float32)
    mean = torch.tensor(in_norm.params["mean"], dtype=torch.float32)

    def f(x):
        x_norm = (x - mean) / std
        return model(x_norm.unsqueeze(0)).squeeze()

    x_t = torch.tensor(x_raw, dtype=torch.float32)
    H = hessian(f, x_t)
    return H.detach().numpy()


def _forward_pass(layers, weights, x_norm):
    """Run forward pass, returning activations and derivatives per layer."""
    dacts = []
    d2acts = []
    h = x_norm.copy()
    for layer in layers:
        w = weights[layer.weights]
        b = weights[layer.bias]
        a = w @ h + b

        act = layer.activation
        if act == "softplus":
            z = np.log(1.0 + np.exp(a))
            da = 1.0 / (1.0 + np.exp(-a))
            d2a = da * (1.0 - da)
        elif act == "tanh":
            z = np.tanh(a)
            da = 1.0 - z**2
            d2a = -2.0 * z * da
        elif act == "sigmoid":
            z = 1.0 / (1.0 + np.exp(-a))
            da = z * (1.0 - z)
            d2a = da * (1.0 - 2.0 * z)
        elif act == "relu":
            z = np.maximum(0.0, a)
            da = (a > 0).astype(float)
            d2a = np.zeros_like(a)
        else:  # identity
            z = a.copy()
            da = np.ones_like(a)
            d2a = np.zeros_like(a)

        dacts.append(da)
        d2acts.append(d2a)
        h = z
    return dacts, d2acts


def _emitter_hessian(model, in_norm, energy_norm, x_raw):
    """Compute d²W/dI² by manually executing the emitter's analytical Hessian logic in Python."""
    exported = extract_weights(model, in_norm, energy_norm)
    layers = exported.layers
    weights = exported.weights
    in_std = exported.input_normalizer["std"]
    in_mean = exported.input_normalizer["mean"]

    # Normalize input and run forward pass
    x_norm = (x_raw - in_mean) / in_std
    dacts, d2acts = _forward_pass(layers, weights, x_norm)

    n_layers = len(layers)
    in_dim = len(x_norm)

    # Backward pass (delta = dW/da)
    deltas = [None] * n_layers
    last = n_layers - 1
    deltas[last] = np.zeros(len(dacts[last]))
    deltas[last][0] = dacts[last][0]  # scalar output

    for i in range(last - 1, -1, -1):
        w_next = weights[layers[i + 1].weights]
        deltas[i] = (w_next.T @ deltas[i + 1]) * dacts[i]

    # Jacobian propagation: P_i (pre-activation), J_i (post-activation)
    Ps = []
    Js = []
    w0 = weights[layers[0].weights]
    P0 = w0.copy()
    J0 = np.diag(dacts[0]) @ w0
    Ps.append(P0)
    Js.append(J0)

    for i in range(1, n_layers):
        wi = weights[layers[i].weights]
        Pi = wi @ Js[i - 1]
        Ji = np.diag(dacts[i]) @ Pi
        Ps.append(Pi)
        Js.append(Ji)

    # Hessian accumulation
    d2W_dx2 = np.zeros((in_dim, in_dim))
    for i in range(n_layers):
        act = layers[i].activation
        if act in ("relu", "identity"):
            continue
        for j in range(len(dacts[i])):
            coeff = deltas[i][j] / dacts[i][j] * d2acts[i][j]
            d2W_dx2 += coeff * np.outer(Ps[i][j, :], Ps[i][j, :])

    # Convert to raw invariant space
    d2W_dI2 = d2W_dx2 / np.outer(in_std, in_std)
    return d2W_dI2


@pytest.mark.parametrize("activation", ["softplus", "tanh", "sigmoid"])
def test_hessian_matches_pytorch(activation):
    """Compare analytical Hessian against PyTorch autograd for smooth activations."""
    model, in_norm, energy_norm = _make_model(activation=activation, hidden_dims=[8, 8])
    rng = np.random.default_rng(123)
    x_raw = rng.standard_normal(3).astype(np.float64)

    H_torch = _pytorch_hessian(model, in_norm, x_raw)
    H_analytical = _emitter_hessian(model, in_norm, energy_norm, x_raw)

    np.testing.assert_allclose(H_analytical, H_torch, atol=1e-5, rtol=1e-4)


def test_hessian_symmetry():
    """Analytical Hessian should be symmetric."""
    model, in_norm, energy_norm = _make_model(activation="softplus")
    x_raw = np.array([3.1, 3.0, 1.0])
    H = _emitter_hessian(model, in_norm, energy_norm, x_raw)
    np.testing.assert_allclose(H, H.T, atol=1e-12)


def test_hessian_relu_is_zero():
    """ReLU has zero second derivative, so Hessian contribution should be zero."""
    model, in_norm, energy_norm = _make_model(activation="relu")
    x_raw = np.array([3.1, 3.0, 1.0])
    H = _emitter_hessian(model, in_norm, energy_norm, x_raw)
    np.testing.assert_allclose(H, 0.0, atol=1e-12)


@pytest.mark.parametrize("hidden_dims", [[16], [8, 8, 8], [4, 8, 4]])
def test_hessian_various_architectures(hidden_dims):
    """Analytical Hessian matches PyTorch for different layer configurations."""
    model, in_norm, energy_norm = _make_model(activation="softplus", hidden_dims=hidden_dims)
    rng = np.random.default_rng(456)
    x_raw = rng.standard_normal(3).astype(np.float64)

    H_torch = _pytorch_hessian(model, in_norm, x_raw)
    H_analytical = _emitter_hessian(model, in_norm, energy_norm, x_raw)

    np.testing.assert_allclose(H_analytical, H_torch, atol=1e-5, rtol=1e-4)


def test_d2act_softplus():
    """Verify softplus second derivative formula: dact * (1 - dact)."""
    a = np.linspace(-3, 3, 100)
    dact = 1.0 / (1.0 + np.exp(-a))  # sigmoid
    d2act = dact * (1.0 - dact)
    # Compare with numerical second derivative of softplus
    h = 1e-5
    sp = lambda x: np.log(1.0 + np.exp(x))
    d2_num = (sp(a + h) - 2 * sp(a) + sp(a - h)) / h**2
    np.testing.assert_allclose(d2act, d2_num, atol=1e-4)


def test_d2act_tanh():
    """Verify tanh second derivative formula: -2 * z * dact."""
    a = np.linspace(-3, 3, 100)
    z = np.tanh(a)
    dact = 1.0 - z**2
    d2act = -2.0 * z * dact
    h = 1e-5
    d2_num = (np.tanh(a + h) - 2 * np.tanh(a) + np.tanh(a - h)) / h**2
    np.testing.assert_allclose(d2act, d2_num, atol=1e-4)


def test_d2act_sigmoid():
    """Verify sigmoid second derivative formula: dact * (1 - 2*z)."""
    a = np.linspace(-3, 3, 100)
    z = 1.0 / (1.0 + np.exp(-a))
    dact = z * (1.0 - z)
    d2act = dact * (1.0 - 2.0 * z)
    h = 1e-5
    sig = lambda x: 1.0 / (1.0 + np.exp(-x))
    d2_num = (sig(a + h) - 2 * sig(a) + sig(a - h)) / h**2
    np.testing.assert_allclose(d2act, d2_num, atol=1e-4)


def test_generated_fortran_has_d2act_no_eps_fd():
    """Generated Fortran should contain d2act but not eps_fd."""
    model, in_norm, energy_norm = _make_model(activation="softplus")
    exported = extract_weights(model, in_norm, energy_norm)
    code = HybridUMATEmitter(exported).emit()
    assert "d2act" in code
    assert "d2W_dI2" in code
    assert "eps_fd" not in code
    assert "nn_input_p" not in code
