import pytest

torch = pytest.importorskip("torch")
from hyper_surrogate.training.losses import EnergyStressLoss, StressLoss, StressTangentLoss  # noqa: E402


def test_stress_loss():
    loss_fn = StressLoss()
    pred = torch.randn(10, 6)
    target = torch.randn(10, 6)
    loss = loss_fn(pred, target)
    assert loss.shape == ()
    assert loss.item() >= 0


def test_stress_tangent_loss():
    loss_fn = StressTangentLoss(alpha=1.0, beta=0.1)
    pred = torch.randn(10, 27)  # 6 stress + 21 tangent
    target = torch.randn(10, 27)  # same layout
    loss = loss_fn(pred, target)
    assert loss.shape == ()
    assert loss.item() >= 0


def test_energy_stress_loss():
    loss_fn = EnergyStressLoss(alpha=1.0, beta=1.0)
    # Simulate ICNN: input -> scalar energy
    x = torch.randn(10, 3, requires_grad=True)
    w_pred = (x**2).sum(dim=1, keepdim=True)  # simple convex function
    w_true = torch.randn(10, 1)
    s_true = torch.randn(10, 3)
    loss = loss_fn(w_pred, w_true, x, s_true)
    assert loss.shape == ()
    assert loss.item() >= 0
