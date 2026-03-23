import numpy as np
import pytest

torch = pytest.importorskip("torch")
from hyper_surrogate.data.dataset import Normalizer  # noqa: E402
from hyper_surrogate.export.fortran.hybrid import HybridUMATEmitter  # noqa: E402
from hyper_surrogate.export.weights import extract_weights  # noqa: E402
from hyper_surrogate.models.mlp import MLP  # noqa: E402


def _make_scalar_model(activation="softplus", hidden_dims=None, input_dim=3):
    """Create a scalar-output MLP with input normalizer (typical SEF setup)."""
    if hidden_dims is None:
        hidden_dims = [8, 8]
    model = MLP(input_dim=input_dim, output_dim=1, hidden_dims=hidden_dims, activation=activation)
    in_norm = Normalizer().fit(np.random.randn(50, input_dim))
    energy_norm = Normalizer().fit(np.random.randn(50, 1))
    return extract_weights(model, in_norm, energy_norm)


def test_rejects_unsupported_architecture():
    exported = _make_scalar_model()
    exported.metadata["architecture"] = "icnn"
    with pytest.raises(ValueError, match="supports"):
        HybridUMATEmitter(exported)


def test_rejects_non_scalar_output():
    model = MLP(input_dim=3, output_dim=6, hidden_dims=[8])
    exported = extract_weights(model)
    with pytest.raises(ValueError, match="scalar"):
        HybridUMATEmitter(exported)


def test_emit_produces_umat():
    exported = _make_scalar_model()
    code = HybridUMATEmitter(exported).emit()
    assert "MODULE nn_sef" in code
    assert "SUBROUTINE nn_eval" in code
    assert "SUBROUTINE umat" in code
    assert "END MODULE" in code
    assert "END SUBROUTINE umat" in code


def test_emit_contains_kinematics():
    exported = _make_scalar_model()
    code = HybridUMATEmitter(exported).emit()
    # Check for analytical kinematics components
    assert "Right Cauchy-Green" in code
    assert "PK2" in code or "pk2" in code.lower()
    assert "Cauchy" in code
    assert "Jaumann" in code


def test_emit_contains_nn_parameters():
    exported = _make_scalar_model()
    code = HybridUMATEmitter(exported).emit()
    assert "w0" in code
    assert "b0" in code
    assert "in_mean" in code
    assert "in_std" in code


def test_emit_activations():
    for act in ["tanh", "relu", "sigmoid", "softplus"]:
        exported = _make_scalar_model(activation=act)
        code = HybridUMATEmitter(exported).emit()
        assert "SUBROUTINE nn_eval" in code


def test_emit_analytical_hessian():
    exported = _make_scalar_model()
    code = HybridUMATEmitter(exported).emit()
    # Check for analytical Hessian (no FD)
    assert "eps_fd" not in code
    assert "d2W_dI2" in code
    assert "d2act" in code
    assert "d2W_dx2" in code


def test_write_to_file(tmp_path):
    exported = _make_scalar_model()
    path = tmp_path / "hybrid_umat.f90"
    HybridUMATEmitter(exported).write(str(path))
    assert path.exists()
    content = path.read_text()
    assert "MODULE nn_sef" in content
    assert "SUBROUTINE umat" in content


# --- Anisotropic (input_dim=5) tests ---


def test_emit_aniso_produces_umat():
    exported = _make_scalar_model(input_dim=5)
    code = HybridUMATEmitter(exported).emit()
    assert "MODULE nn_sef" in code
    assert "SUBROUTINE nn_eval" in code
    assert "SUBROUTINE umat" in code


def test_emit_aniso_nn_eval_signature():
    exported = _make_scalar_model(input_dim=5)
    code = HybridUMATEmitter(exported).emit()
    assert "nn_input(5)" in code
    assert "dW_dI(5)" in code
    assert "d2W_dI2(5,5)" in code


def test_emit_aniso_fiber_invariants():
    exported = _make_scalar_model(input_dim=5)
    code = HybridUMATEmitter(exported).emit()
    # Fiber direction from props
    assert "a0_1(1) = props(1)" in code
    # I4, I5 computation
    assert "I4_1 = I4_1 + a0_1(ii) * Ca0_1(ii)" in code
    assert "I5_1 = I5_1 + Ca0_1(ii) * Ca0_1(ii)" in code
    # NN input includes fiber invariants
    assert "nn_input(4) = I4_1" in code
    assert "nn_input(5) = I5_1" in code


def test_emit_aniso_fiber_derivatives():
    exported = _make_scalar_model(input_dim=5)
    code = HybridUMATEmitter(exported).emit()
    # dI4/dC = a0 x a0
    assert "dI4_1_dC(ii,jj) = a0_1(ii) * a0_1(jj)" in code
    # dI5/dC = a0 x Ca0 + Ca0 x a0
    assert "dI5_1_dC(ii,jj) = a0_1(ii) * Ca0_1(jj) + Ca0_1(ii) * a0_1(jj)" in code
    # PK2 includes fiber terms
    assert "dW_dI(4)" in code
    assert "dW_dI(5)" in code


def test_emit_aniso_tangent():
    exported = _make_scalar_model(input_dim=5)
    code = HybridUMATEmitter(exported).emit()
    # dIdC array has 5 slots
    assert "dIdC(3, 3, 5)" in code
    # Loops over 5 invariants for tangent
    assert "DO mm = 1, 5" in code
    assert "DO nn = 1, 5" in code
    # d²I5/dCdC contribution present
    assert "a0_1(ii)*a0_1(ll)*eye3(jj,kk)" in code


def test_emit_aniso_header():
    exported = _make_scalar_model(input_dim=5)
    code = HybridUMATEmitter(exported).emit()
    assert "I1_bar, I2_bar, J, I4_1, I5_1" in code
    assert "a0_1 from props(1:3)" in code


def test_emit_isotropic_no_fiber():
    """Isotropic (in_dim=3) should NOT contain fiber invariant code."""
    exported = _make_scalar_model(input_dim=3)
    code = HybridUMATEmitter(exported).emit()
    assert "dI4_1_dC" not in code
    assert "dI5_1_dC" not in code
    assert "props(1)" not in code
    assert "nn_input(3)" in code
    assert "dIdC(3, 3, 3)" in code
