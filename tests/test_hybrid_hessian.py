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
from hyper_surrogate.models.polyconvex import PolyconvexICNN  # noqa: E402


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


def _make_polyconvex_model(groups, hidden_dims=None):
    if hidden_dims is None:
        hidden_dims = [8, 8]
    in_dim = max(idx for g in groups for idx in g) + 1
    model = PolyconvexICNN(groups=groups, hidden_dims=hidden_dims, activation="softplus")
    in_norm = Normalizer().fit(np.random.default_rng(42).standard_normal((50, in_dim)))
    energy_norm = Normalizer().fit(np.random.default_rng(42).standard_normal((50, 1)))
    return model, in_norm, energy_norm


def _pytorch_polyconvex_hessian(model, in_norm, x_raw):
    model.eval()
    std = torch.tensor(in_norm.params["std"], dtype=torch.float32)
    mean = torch.tensor(in_norm.params["mean"], dtype=torch.float32)

    def f(x):
        x_norm = (x - mean) / std
        return model(x_norm.unsqueeze(0)).squeeze()

    x_t = torch.tensor(x_raw, dtype=torch.float32)
    return hessian(f, x_t).detach().numpy()


def _polyconvex_emitter_hessian(model, in_norm, energy_norm, x_raw):
    """Mirror hybrid.py:_emit_poly_nn_forward_and_backward (lines 416-592) in numpy."""
    exported = extract_weights(model, in_norm, energy_norm)
    weights = exported.weights
    branches = exported.metadata["branches"]
    in_std = exported.input_normalizer["std"]
    in_mean = exported.input_normalizer["mean"]

    x_norm = (x_raw - in_mean) / in_std
    in_dim = len(x_norm)
    d2W_dI2 = np.zeros((in_dim, in_dim))

    def _softplus(w):
        return np.log(1.0 + np.exp(w))

    for bi, branch in enumerate(branches):
        b_layers = branch["layers"]
        indices = branch["input_indices"]
        b_in = len(indices)
        n_hidden = len(b_layers) - 1  # last entry is the identity output layer
        prefix = f"branches.{bi}."
        xb = x_norm[indices]

        # Forward pass (softplus everywhere on hidden layers)
        a_arr, dact_arr, d2act_arr, z_arr = [], [], [], []

        # Layer 0: x-path only
        W0 = weights[b_layers[0]["weights"]]
        b0 = weights[b_layers[0]["bias"]]
        a0 = W0 @ xb + b0
        z0 = _softplus(a0)
        da0 = 1.0 / (1.0 + np.exp(-a0))
        a_arr.append(a0)
        z_arr.append(z0)
        dact_arr.append(da0)
        d2act_arr.append(da0 * (1.0 - da0))

        # Hidden layers 1..n_hidden-1: wz (softplus on raw) + wx skip
        for li in range(1, n_hidden):
            wz = _softplus(weights[b_layers[li]["weights"]])
            wx = weights[prefix + f"wx_layers.{li}.weight"]
            bli = weights[b_layers[li]["bias"]]
            ali = wz @ z_arr[li - 1] + wx @ xb + bli
            zli = _softplus(ali)
            dali = 1.0 / (1.0 + np.exp(-ali))
            a_arr.append(ali)
            z_arr.append(zli)
            dact_arr.append(dali)
            d2act_arr.append(dali * (1.0 - dali))

        # Output (identity, linear): only need wz_out for backward init
        last_hidden = n_hidden - 1
        wz_out = _softplus(weights[b_layers[n_hidden]["weights"]])  # shape (1, hidden)

        # Backward pass: deltas[li] = ∂Ψ_branch / ∂a^(li)
        deltas: list = [None] * n_hidden
        deltas[last_hidden] = wz_out[0, :] * dact_arr[last_hidden]
        for li in range(last_hidden - 1, -1, -1):
            wz_next = _softplus(weights[b_layers[li + 1]["weights"]])
            deltas[li] = (wz_next.T @ deltas[li + 1]) * dact_arr[li]

        # Jacobian propagation: P^(l) = d a^(l)/d xb (with ICNN skip on layer >= 1)
        Ps, Js = [], []
        P0 = W0.copy()
        Ps.append(P0)
        Js.append(np.diag(dact_arr[0]) @ P0)
        for li in range(1, n_hidden):
            wz = _softplus(weights[b_layers[li]["weights"]])
            wx = weights[prefix + f"wx_layers.{li}.weight"]
            Pli = wz @ Js[li - 1] + wx
            Ps.append(Pli)
            Js.append(np.diag(dact_arr[li]) @ Pli)

        # Per-branch Hessian
        d2W_b = np.zeros((b_in, b_in))
        for li in range(n_hidden):
            for j in range(len(dact_arr[li])):
                coeff = deltas[li][j] / dact_arr[li][j] * d2act_arr[li][j]
                d2W_b += coeff * np.outer(Ps[li][j, :], Ps[li][j, :])

        # Scatter (block at indices, divide by std*std)
        for si, idx_i in enumerate(indices):
            for sj, idx_j in enumerate(indices):
                d2W_dI2[idx_i, idx_j] += d2W_b[si, sj] / (in_std[idx_i] * in_std[idx_j])

    return d2W_dI2


@pytest.mark.parametrize(
    "groups",
    [
        [[0], [1], [2]],  # isotropic 3-group (I1, I2, J)
        [[0], [1], [2], [3, 4]],  # anisotropic 4-group with 2-D fiber subspace
    ],
)
def test_polyconvex_hessian_matches_pytorch(groups):
    """Polyconvex emitter Hessian matches PyTorch autograd on random-init weights."""
    model, in_norm, energy_norm = _make_polyconvex_model(groups, hidden_dims=[8, 8])
    in_dim = max(idx for g in groups for idx in g) + 1
    rng = np.random.default_rng(789)
    x_raw = rng.standard_normal(in_dim).astype(np.float64)

    H_torch = _pytorch_polyconvex_hessian(model, in_norm, x_raw)
    H_analytical = _polyconvex_emitter_hessian(model, in_norm, energy_norm, x_raw)

    np.testing.assert_allclose(H_analytical, H_torch, atol=1e-5, rtol=1e-4)


def test_polyconvex_hessian_singleton_groups_are_diagonal():
    """Singleton groups [[0],[1],[2]] -> Hessian is purely diagonal (no cross-group coupling)."""
    model, in_norm, energy_norm = _make_polyconvex_model([[0], [1], [2]], hidden_dims=[8, 8])
    rng = np.random.default_rng(321)
    x_raw = rng.standard_normal(3).astype(np.float64)

    H = _polyconvex_emitter_hessian(model, in_norm, energy_norm, x_raw)
    off_diag = H - np.diag(np.diag(H))
    np.testing.assert_allclose(off_diag, 0.0, atol=1e-12)


def test_polyconvex_hessian_cross_group_off_diagonals_are_zero():
    """For groups [[0],[1],[2],[3,4]], H[i,j] must be zero when i,j fall in different groups."""
    groups = [[0], [1], [2], [3, 4]]
    model, in_norm, energy_norm = _make_polyconvex_model(groups, hidden_dims=[8, 8])
    rng = np.random.default_rng(654)
    x_raw = rng.standard_normal(5).astype(np.float64)

    H = _polyconvex_emitter_hessian(model, in_norm, energy_norm, x_raw)
    cross_pairs = [(i, j) for i in [0, 1, 2] for j in [3, 4]]
    for i, j in cross_pairs:
        assert abs(H[i, j]) < 1e-12 and abs(H[j, i]) < 1e-12, f"H[{i},{j}]={H[i, j]} not zero"


@pytest.mark.parametrize("hidden_dims", [[16], [4, 4, 4]])
def test_polyconvex_hessian_various_depths(hidden_dims):
    """Polyconvex Hessian matches PyTorch across branch depths."""
    groups = [[0], [1], [2]]
    model, in_norm, energy_norm = _make_polyconvex_model(groups, hidden_dims=hidden_dims)
    rng = np.random.default_rng(987)
    x_raw = rng.standard_normal(3).astype(np.float64)

    H_torch = _pytorch_polyconvex_hessian(model, in_norm, x_raw)
    H_analytical = _polyconvex_emitter_hessian(model, in_norm, energy_norm, x_raw)

    np.testing.assert_allclose(H_analytical, H_torch, atol=1e-5, rtol=1e-4)


def _make_mlp_with_input_dim(input_dim, activation="softplus", hidden_dims=None):
    if hidden_dims is None:
        hidden_dims = [8, 8]
    model = MLP(input_dim=input_dim, output_dim=1, hidden_dims=hidden_dims, activation=activation)
    in_norm = Normalizer().fit(np.random.default_rng(42).standard_normal((50, input_dim)))
    energy_norm = Normalizer().fit(np.random.default_rng(42).standard_normal((50, 1)))
    return model, in_norm, energy_norm


def _pytorch_hessian_anyD(model, in_norm, x_raw):
    model.eval()
    std = torch.tensor(in_norm.params["std"], dtype=torch.float32)
    mean = torch.tensor(in_norm.params["mean"], dtype=torch.float32)

    def f(x):
        x_norm = (x - mean) / std
        return model(x_norm.unsqueeze(0)).squeeze()

    return hessian(f, torch.tensor(x_raw, dtype=torch.float32)).detach().numpy()


@pytest.mark.parametrize("input_dim", [3, 5])
def test_mlp_hessian_matches_pytorch_input_dim(input_dim):
    """MLP Hessian matches PyTorch for both isotropic (3) and anisotropic (5) input shapes."""
    model, in_norm, energy_norm = _make_mlp_with_input_dim(input_dim, hidden_dims=[8, 8])
    rng = np.random.default_rng(2024)
    x_raw = rng.standard_normal(input_dim).astype(np.float64)

    H_torch = _pytorch_hessian_anyD(model, in_norm, x_raw)
    H_analytical = _emitter_hessian(model, in_norm, energy_norm, x_raw)

    np.testing.assert_allclose(H_analytical, H_torch, atol=1e-5, rtol=1e-4)


def test_mlp_hessian_trained_at_saturated_activations():
    """After brief training, Hessian still matches PyTorch at points spanning the training domain.

    The codegen's `delta/dact * d2act` division has its largest numerical exposure when softplus
    pre-activations saturate (small dact). This test exercises that regime on trained weights.
    """
    torch.manual_seed(0)
    np.random.seed(0)
    input_dim = 3
    model = MLP(input_dim=input_dim, output_dim=1, hidden_dims=[16, 16], activation="softplus")

    # Synthetic Neo-Hookean-like target: W = 0.5 * (I1 - 3) + 50 * (J - 1)^2 on (I1, I2, J).
    rng = np.random.default_rng(0)
    n = 256
    x_train = rng.uniform(low=[2.7, 2.7, 0.85], high=[3.5, 3.5, 1.15], size=(n, 3)).astype(np.float32)
    w_train = (0.5 * (x_train[:, 0:1] - 3.0) + 50.0 * (x_train[:, 2:3] - 1.0) ** 2).astype(np.float32)

    in_norm = Normalizer().fit(x_train)
    energy_norm = Normalizer().fit(w_train)

    x_t = torch.tensor((x_train - in_norm.params["mean"]) / in_norm.params["std"])
    w_t = torch.tensor(w_train)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(50):
        opt.zero_grad()
        loss = ((model(x_t) - w_t) ** 2).mean()
        loss.backward()
        opt.step()

    probe_points = [
        np.array([2.7, 2.7, 0.85], dtype=np.float64),  # corner of training domain
        np.array([3.5, 3.5, 1.15], dtype=np.float64),  # opposite corner
        np.array([3.1, 3.0, 1.00], dtype=np.float64),  # near reference (I=identity)
    ]
    for x_raw in probe_points:
        H_torch = _pytorch_hessian_anyD(model, in_norm, x_raw)
        H_analytical = _emitter_hessian(model, in_norm, energy_norm, x_raw)
        np.testing.assert_allclose(H_analytical, H_torch, atol=1e-5, rtol=1e-4)


def test_reference_offset_makes_energy_zero_at_C_eq_I():
    """The hybrid emitter subtracts Psi(reference) so Psi(C=I)=0 exactly.

    Without the offset, the trained model emits a small residual energy at
    the reference configuration; the emitter pre-computes this value and
    embeds the subtraction as W_REF_OFFSET. Verifies the numpy mirror of
    the Fortran path returns ~0 at the reference point.
    """
    model, in_norm, energy_norm = _make_model(activation="softplus", hidden_dims=[8, 8])
    exported = extract_weights(model, in_norm, energy_norm)
    emitter = HybridUMATEmitter(exported, enforce_stress_free_reference=True)
    code = emitter.emit()
    assert "W_REF_OFFSET" in code, "emitted UMAT must declare the reference-offset PARAMETER"
    assert "W_nn = z" in code and "- W_REF_OFFSET" in code, "W_nn output must subtract W_REF_OFFSET"

    # Verify the numerical offset matches the model's prediction at C=I.
    in_dim = exported.metadata["input_dim"]
    x_ref = np.array([3.0, 3.0, 1.0] + [1.0] * (in_dim - 3))
    mean = np.asarray(in_norm.params["mean"], dtype=float)
    std = np.asarray(in_norm.params["std"], dtype=float)
    x_norm = (x_ref - mean) / std
    x_t = torch.tensor(x_norm, dtype=torch.float32).unsqueeze(0)
    model.eval()
    W_ref_torch = float(model(x_t).detach().numpy().flatten()[0])
    np.testing.assert_allclose(emitter._W_ref, W_ref_torch, atol=1e-5, rtol=1e-4)

    # And confirm the offset can be disabled.
    emitter_off = HybridUMATEmitter(exported, enforce_stress_free_reference=False)
    assert emitter_off._W_ref == 0.0
    assert "W_REF_OFFSET = 0.0" in emitter_off.emit() or "0.000000000000000e+00" in emitter_off.emit()


def test_reference_gradient_makes_stress_zero_at_C_eq_I():
    """The hybrid emitter subtracts dW/dI(C=I) for non-deviatoric invariants
    so that sigma(C=I) = 0 exactly at deployment.

    The deviatoric invariants Ī₁, Ī₂ have dI/dC|_{C=I} = 0, so the trained
    gradient on those components never reaches sigma at the reference and
    is left untouched (the corresponding entries of dW_dI_REF are masked
    to zero).  J (and any fiber invariants) have dI/dC|_{C=I} ≠ 0, so the
    trained gradient there directly biases sigma(C=I).  The emitter
    embeds dW/dI at reference as a PARAMETER array and the Fortran
    subtracts it from the runtime gradient.  This test verifies:

    1. dW_dI_REF is present in the emitted code and used in the subtraction;
    2. components 0 and 1 (Ī₁, Ī₂) of dW_dI_REF are exactly zero (masked);
    3. component 2+ (J, optional I4, I5) match the model's trained
       gradient at the reference (computed by autograd);
    4. with the correction enabled, the numpy-mirror chain rule yields
       sigma(C=I) = 0 to floating-point precision;
    5. the correction can be disabled via enforce_stress_free_reference=False.
    """
    model, in_norm, energy_norm = _make_model(activation="softplus", hidden_dims=[8, 8])
    exported = extract_weights(model, in_norm, energy_norm)
    emitter = HybridUMATEmitter(exported, enforce_stress_free_reference=True)
    code = emitter.emit()

    # 1. Fortran has dW_dI_REF and the subtraction.
    assert "dW_dI_REF" in code, "emitted UMAT must declare dW_dI_REF PARAMETER"
    assert "dW_dI(1) = dW_dI(1) - dW_dI_REF(1)" in code, "runtime subtraction missing"

    in_dim = exported.metadata["input_dim"]
    dW_dI_ref = np.asarray(emitter._dW_dI_ref, dtype=float)
    assert dW_dI_ref.shape == (in_dim,)

    # 2. Deviatoric components must be exactly zero (the mask).
    np.testing.assert_array_equal(dW_dI_ref[:2], 0.0)

    # 3. Component 2+ must match the trained model's autograd gradient at
    #    reference (in raw-invariant space, i.e. divided by in_std).
    x_ref = np.array([3.0, 3.0, 1.0] + [1.0] * (in_dim - 3))
    mean = np.asarray(in_norm.params["mean"], dtype=float)
    std = np.asarray(in_norm.params["std"], dtype=float)
    x_norm = (x_ref - mean) / std
    model.eval()
    x_t = torch.tensor(x_norm, dtype=torch.float32, requires_grad=True).unsqueeze(0)
    W = model(x_t)
    dW_dx_norm = torch.autograd.grad(W.sum(), x_t)[0].detach().numpy().flatten()
    dW_dI_autograd = dW_dx_norm / std
    np.testing.assert_allclose(
        dW_dI_ref[2:],
        dW_dI_autograd[2:],
        atol=1e-4,
        rtol=1e-3,
        err_msg="dW_dI_REF[2:] must equal trained autograd gradient on J/fiber invariants",
    )

    # 4. Numpy mirror of the Fortran chain rule at C=I yields sigma = 0.
    #    Only dJ/dC|_{C=I} = 0.5 I is non-zero; the deviatoric chain-rule
    #    weights vanish.  Pushing the corrected dW/dJ through gives sigma ~ 0.
    eye = np.eye(3)
    dJ_dC = 0.5 * eye
    S = 2.0 * (dW_dI_autograd[2] - dW_dI_ref[2]) * dJ_dC
    sigma = eye @ S @ eye.T
    # Residual comes from the finite-difference precision of
    # _compute_reference_gradient (eps=1e-5 -> ~1e-9 residual on dW/dJ);
    # ~10 orders of magnitude below typical Cauchy stress in MPa.
    np.testing.assert_allclose(sigma, 0.0, atol=1e-8, err_msg="sigma(C=I) must be near-zero after dW_dI_REF correction")

    # 5. Disabling the correction zeroes dW_dI_REF.
    emitter_off = HybridUMATEmitter(exported, enforce_stress_free_reference=False)
    np.testing.assert_array_equal(emitter_off._dW_dI_ref, np.zeros(in_dim))


def test_polyconvex_emits_dW_dI_REF_inside_module():
    """Polyconvex emitter declares dW_dI_REF inside the nn_sef module so
    that the subsequent `dW_dI(k) = dW_dI(k) - dW_dI_REF(k)` lines (also
    emitted inside the module's nn_eval body) reference an in-scope
    PARAMETER.  Regression for an emitter bug where the MLP variant of
    `_emit_..._nn_parameters` declared dW_dI_REF but the polyconvex
    variant did not, causing gfortran to error with
    `Function 'dw_di_ref' has no IMPLICIT type` on the subtraction lines.
    """
    rng = np.random.default_rng(0)
    in_norm = Normalizer().fit(rng.standard_normal((50, 3)))
    energy_norm = Normalizer().fit(rng.standard_normal((50, 1)))
    poly = PolyconvexICNN(groups=[[0], [1], [2]], hidden_dims=[4, 4], activation="softplus")
    exported = extract_weights(poly, in_norm, energy_norm)
    emitter = HybridUMATEmitter(exported, enforce_stress_free_reference=True)
    code = emitter.emit()

    # Locate the nn_sef module boundary and the subtraction lines.
    lines = code.splitlines()
    mod_start = next(i for i, ln in enumerate(lines) if ln.strip().startswith("MODULE nn_sef"))
    mod_end = next(i for i, ln in enumerate(lines) if ln.strip().startswith("END MODULE nn_sef"))
    decl = next(i for i, ln in enumerate(lines) if "PARAMETER :: dW_dI_REF" in ln)
    subs = [i for i, ln in enumerate(lines) if "dW_dI(1) = dW_dI(1) - dW_dI_REF(1)" in ln]
    assert subs, "polyconvex emitter must emit dW_dI(k) - dW_dI_REF(k) lines"
    assert mod_start < decl < mod_end, "dW_dI_REF declaration must be inside MODULE nn_sef"
    for s in subs:
        assert mod_start < s < mod_end, "dW_dI_REF subtraction must be inside MODULE nn_sef"


def test_generated_fortran_has_d2act_no_eps_fd():
    """Generated Fortran should contain d2act but not eps_fd."""
    model, in_norm, energy_norm = _make_model(activation="softplus")
    exported = extract_weights(model, in_norm, energy_norm)
    code = HybridUMATEmitter(exported).emit()
    assert "d2act" in code
    assert "d2W_dI2" in code
    assert "eps_fd" not in code
    assert "nn_input_p" not in code
