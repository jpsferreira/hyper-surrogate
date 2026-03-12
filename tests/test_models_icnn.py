import pytest

torch = pytest.importorskip("torch")
from hyper_surrogate.models.icnn import ICNN  # noqa: E402


def test_icnn_forward_scalar_output():
    model = ICNN(input_dim=3, hidden_dims=[32, 32])
    x = torch.randn(10, 3)
    y = model(x)
    assert y.shape == (10, 1)


def test_icnn_convexity():
    """Output should be convex w.r.t. input: f(tx1 + (1-t)x2) <= t*f(x1) + (1-t)*f(x2)."""
    model = ICNN(input_dim=3, hidden_dims=[32, 32])
    model.eval()
    x1 = torch.randn(50, 3)
    x2 = torch.randn(50, 3)
    t = 0.5
    with torch.no_grad():
        f_mix = model(t * x1 + (1 - t) * x2)
        f_avg = t * model(x1) + (1 - t) * model(x2)
    # Convexity: f(mix) <= f(avg) (with numerical tolerance)
    assert (f_mix <= f_avg + 1e-5).all()


def test_icnn_gradient():
    """Can compute gradient of output w.r.t. input."""
    model = ICNN(input_dim=3, hidden_dims=[16])
    x = torch.randn(5, 3, requires_grad=True)
    y = model(x)
    grad = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
    assert grad.shape == (5, 3)


def test_icnn_layer_sequence():
    model = ICNN(input_dim=3, hidden_dims=[32, 16])
    seq = model.layer_sequence()
    assert len(seq) > 0
    # Should have both wz and wx type entries
    keys = [s.weights for s in seq]
    assert any("wz" in k for k in keys)
    assert any("wx" in k for k in keys)


def test_icnn_export_weights():
    model = ICNN(input_dim=3, hidden_dims=[16])
    weights = model.export_weights()
    assert len(weights) > 0


def test_icnn_properties():
    model = ICNN(input_dim=3)
    assert model.input_dim == 3
    assert model.output_dim == 1
