import os
import tempfile

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from hyper_surrogate.data.dataset import Normalizer  # noqa: E402
from hyper_surrogate.export.weights import ExportedModel, extract_weights  # noqa: E402
from hyper_surrogate.models.mlp import MLP  # noqa: E402


def test_extract_weights_mlp():
    model = MLP(input_dim=3, output_dim=6, hidden_dims=[16])
    exported = extract_weights(model)
    assert isinstance(exported, ExportedModel)
    assert exported.metadata["architecture"] == "mlp"
    assert exported.metadata["input_dim"] == 3
    assert exported.metadata["output_dim"] == 6
    assert len(exported.layers) > 0
    assert len(exported.weights) > 0


def test_extract_with_normalizers():
    model = MLP(input_dim=3, output_dim=6, hidden_dims=[16])
    in_norm = Normalizer().fit(np.random.randn(100, 3))
    out_norm = Normalizer().fit(np.random.randn(100, 6))
    exported = extract_weights(model, in_norm, out_norm)
    assert exported.input_normalizer is not None
    assert exported.output_normalizer is not None


def test_save_load_roundtrip():
    model = MLP(input_dim=3, output_dim=6, hidden_dims=[16])
    exported = extract_weights(model)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "model.npz")
        exported.save(path)
        loaded = ExportedModel.load(path)
        assert loaded.metadata["architecture"] == "mlp"
        for k in exported.weights:
            np.testing.assert_array_equal(exported.weights[k], loaded.weights[k])
