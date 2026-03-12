import pytest

torch = pytest.importorskip("torch")
from hyper_surrogate.models.mlp import MLP  # noqa: E402


def test_mlp_forward_shape():
    model = MLP(input_dim=3, output_dim=6, hidden_dims=[32, 32])
    x = torch.randn(10, 3)
    y = model(x)
    assert y.shape == (10, 6)


def test_mlp_layer_sequence():
    model = MLP(input_dim=3, output_dim=6, hidden_dims=[32, 16])
    seq = model.layer_sequence()
    assert len(seq) == 3  # 2 hidden + 1 output
    assert seq[0].activation == "tanh"
    assert seq[-1].activation == "identity"


def test_mlp_export_weights():
    model = MLP(input_dim=3, output_dim=6, hidden_dims=[32])
    weights = model.export_weights()
    assert "layers.0.weight" in weights
    assert "layers.0.bias" in weights
    assert weights["layers.0.weight"].shape == (32, 3)


def test_mlp_properties():
    model = MLP(input_dim=3, output_dim=6)
    assert model.input_dim == 3
    assert model.output_dim == 6
