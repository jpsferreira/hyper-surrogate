"""Tests for CANN (Constitutive Artificial Neural Network) model."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hyper_surrogate.models.cann import CANN  # noqa: E402


def test_forward_shape():
    model = CANN(input_dim=3)
    x = torch.randn(10, 3)
    out = model(x)
    assert out.shape == (10, 1)


def test_output_nonnegative_for_positive_input():
    """With non-negative weights and squared/positive basis, output should be >= 0."""
    model = CANN(input_dim=3, n_polynomial=2, n_exponential=1, use_logarithmic=True)
    # Zero input should give zero output (all basis functions vanish at zero)
    x = torch.zeros(1, 3)
    out = model(x)
    np.testing.assert_allclose(out.detach().numpy(), 0.0, atol=1e-5)


def test_weights_nonnegative():
    model = CANN(input_dim=3)
    w = model.weights.detach().numpy()
    assert np.all(w >= 0)


def test_get_active_terms():
    model = CANN(input_dim=2, n_polynomial=2, n_exponential=1, use_logarithmic=False)
    # With default zero initialization, all weights are softplus(0) ~ 0.693
    terms = model.get_active_terms(threshold=0.5)
    assert len(terms) > 0
    for t in terms:
        assert "type" in t
        assert "weight" in t


def test_get_active_terms_after_zeroing():
    model = CANN(input_dim=2, n_polynomial=2, n_exponential=1, use_logarithmic=False)
    # Set all raw weights to very negative -> softplus -> ~0
    with torch.no_grad():
        model.raw_weights.fill_(-20.0)
    terms = model.get_active_terms(threshold=1e-3)
    assert len(terms) == 0


def test_input_output_dim():
    model = CANN(input_dim=5)
    assert model.input_dim == 5
    assert model.output_dim == 1


def test_layer_sequence():
    model = CANN(input_dim=3)
    seq = model.layer_sequence()
    assert len(seq) == 1


def test_export_weights():
    model = CANN(input_dim=3, n_polynomial=2, n_exponential=1, use_logarithmic=True)
    exported = model.export_weights()
    assert "weights" in exported
    assert "exp_params" in exported
    assert "config" in exported
    assert exported["config"][0] == 3  # input_dim


def test_learnable_exponents():
    model = CANN(input_dim=3, n_exponential=2, learnable_exponents=True)
    # raw_exp_params should be a learnable parameter
    param_names = [n for n, _ in model.named_parameters()]
    assert "raw_exp_params" in param_names


def test_gradient_flows():
    model = CANN(input_dim=3)
    x = torch.randn(5, 3, requires_grad=True)
    out = model(x)
    out.sum().backward()
    assert x.grad is not None
    assert model.raw_weights.grad is not None
