import numpy as np
import pytest

torch = pytest.importorskip("torch")
from hyper_surrogate.data.dataset import Normalizer  # noqa: E402
from hyper_surrogate.export.fortran.emitter import FortranEmitter  # noqa: E402
from hyper_surrogate.export.weights import extract_weights  # noqa: E402
from hyper_surrogate.models.mlp import MLP  # noqa: E402


def test_emit_mlp_produces_fortran():
    model = MLP(input_dim=3, output_dim=6, hidden_dims=[8])
    exported = extract_weights(model)
    code = FortranEmitter(exported).emit()
    assert "MODULE nn_surrogate" in code
    assert "SUBROUTINE nn_forward" in code
    assert "MATMUL" in code
    assert "END MODULE" in code


def test_emit_mlp_with_normalizers():
    model = MLP(input_dim=3, output_dim=6, hidden_dims=[8])
    in_norm = Normalizer().fit(np.random.randn(50, 3))
    out_norm = Normalizer().fit(np.random.randn(50, 6))
    exported = extract_weights(model, in_norm, out_norm)
    code = FortranEmitter(exported).emit()
    assert "in_mean" in code.lower() or "input" in code.lower()


def test_emit_mlp_activations():
    for act in ["tanh", "relu", "sigmoid", "softplus"]:
        model = MLP(input_dim=3, output_dim=6, hidden_dims=[8], activation=act)
        exported = extract_weights(model)
        code = FortranEmitter(exported).emit()
        assert "SUBROUTINE nn_forward" in code


def test_write_to_file(tmp_path):
    model = MLP(input_dim=3, output_dim=6, hidden_dims=[8])
    exported = extract_weights(model)
    path = tmp_path / "test.f90"
    FortranEmitter(exported).write(str(path))
    assert path.exists()
    content = path.read_text()
    assert "MODULE nn_surrogate" in content
