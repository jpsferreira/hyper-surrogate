import numpy as np
import pytest

torch = pytest.importorskip("torch")
from hyper_surrogate.data.dataset import Normalizer  # noqa: E402
from hyper_surrogate.export.fortran.hybrid import HybridUMATEmitter  # noqa: E402
from hyper_surrogate.export.weights import extract_weights  # noqa: E402
from hyper_surrogate.models.mlp import MLP  # noqa: E402


def _make_scalar_model(activation="softplus", hidden_dims=None):
    """Create a scalar-output MLP with input normalizer (typical SEF setup)."""
    if hidden_dims is None:
        hidden_dims = [8, 8]
    model = MLP(input_dim=3, output_dim=1, hidden_dims=hidden_dims, activation=activation)
    in_norm = Normalizer().fit(np.random.randn(50, 3))
    energy_norm = Normalizer().fit(np.random.randn(50, 1))
    return extract_weights(model, in_norm, energy_norm)


def test_rejects_non_mlp():
    exported = _make_scalar_model()
    exported.metadata["architecture"] = "icnn"
    with pytest.raises(ValueError, match="only supports MLP"):
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


def test_emit_finite_differences():
    exported = _make_scalar_model()
    code = HybridUMATEmitter(exported).emit()
    # Check for d²W/dI² via FD
    assert "eps_fd" in code
    assert "d2W_dI2" in code


def test_write_to_file(tmp_path):
    exported = _make_scalar_model()
    path = tmp_path / "hybrid_umat.f90"
    HybridUMATEmitter(exported).write(str(path))
    assert path.exists()
    content = path.read_text()
    assert "MODULE nn_sef" in content
    assert "SUBROUTINE umat" in content
