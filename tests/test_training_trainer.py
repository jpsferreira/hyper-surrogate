import numpy as np
import pytest

torch = pytest.importorskip("torch")
from hyper_surrogate.data.dataset import MaterialDataset  # noqa: E402
from hyper_surrogate.models.mlp import MLP  # noqa: E402
from hyper_surrogate.training.losses import StressLoss  # noqa: E402
from hyper_surrogate.training.trainer import Trainer, TrainingResult  # noqa: E402


def _make_datasets():
    """Simple linear regression dataset."""
    rng = np.random.default_rng(42)
    x = rng.standard_normal((200, 3)).astype(np.float32)
    W = rng.standard_normal((3, 6)).astype(np.float32)
    y = (x @ W).astype(np.float32)
    train = MaterialDataset(x[:170], y[:170])
    val = MaterialDataset(x[170:], y[170:])
    return train, val


def test_trainer_fit_returns_result():
    train, val = _make_datasets()
    model = MLP(input_dim=3, output_dim=6, hidden_dims=[16])
    result = Trainer(model, train, val, loss_fn=StressLoss(), max_epochs=5).fit()
    assert isinstance(result, TrainingResult)
    assert len(result.history["train_loss"]) == 5
    assert result.best_epoch >= 0


def test_trainer_loss_decreases():
    train, val = _make_datasets()
    model = MLP(input_dim=3, output_dim=6, hidden_dims=[32, 32])
    result = Trainer(model, train, val, loss_fn=StressLoss(), max_epochs=50, patience=50).fit()
    assert result.history["train_loss"][-1] < result.history["train_loss"][0]
