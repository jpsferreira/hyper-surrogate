"""Tests for benchmarking metrics and reporting."""

import numpy as np
import pytest

from hyper_surrogate.benchmarking.metrics import (
    BenchmarkResult,
    _mae,
    _max_abs_error,
    _r_squared,
    _rmse,
    benchmark_suite,
    benchmark_surrogate,
)
from hyper_surrogate.benchmarking.reporting import results_to_latex, results_to_markdown
from hyper_surrogate.models.mlp import MLP


def test_r_squared_perfect():
    y = np.array([1.0, 2.0, 3.0])
    assert _r_squared(y, y) == 1.0


def test_r_squared_zero():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 2.0, 2.0])  # predicting mean
    np.testing.assert_allclose(_r_squared(y_true, y_pred), 0.0, atol=1e-10)


def test_rmse():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 2.1, 3.1])
    np.testing.assert_allclose(_rmse(y_true, y_pred), 0.1, atol=1e-10)


def test_mae():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 1.9, 3.2])
    # MAE = (0.1 + 0.1 + 0.2) / 3 = 0.1333...
    np.testing.assert_allclose(_mae(y_true, y_pred), 4.0 / 30.0, atol=1e-10)


def test_benchmark_result_summary():
    r = BenchmarkResult(
        material_name="NeoHooke",
        model_name="MLP",
        metrics={"r2": 0.999, "rmse": 0.001},
        n_parameters=5000,
        training_time=12.5,
    )
    s = r.summary()
    assert "NeoHooke" in s
    assert "MLP" in s
    assert "5000" in s


def test_benchmark_result_to_dict():
    r = BenchmarkResult(
        material_name="NeoHooke",
        model_name="MLP",
        metrics={"r2": 0.999},
        n_parameters=5000,
    )
    d = r.to_dict()
    assert d["material"] == "NeoHooke"
    assert d["r2"] == 0.999


def test_results_to_latex():
    results = [
        BenchmarkResult("NeoHooke", "MLP", {"r2": 0.999}, n_parameters=5000, training_time=1.0),
        BenchmarkResult("Yeoh", "ICNN", {"r2": 0.995}, n_parameters=3000, training_time=2.0),
    ]
    tex = results_to_latex(results)
    assert r"\begin{table}" in tex
    assert "NeoHooke" in tex
    assert "Yeoh" in tex


def test_results_to_markdown():
    results = [
        BenchmarkResult("NeoHooke", "MLP", {"r2": 0.999}, n_parameters=5000, training_time=1.0),
    ]
    md = results_to_markdown(results)
    assert "| NeoHooke" in md
    assert "---" in md


def test_empty_results():
    assert results_to_latex([]) == ""
    assert results_to_markdown([]) == ""


def test_max_abs_error():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.5, 3.0])
    np.testing.assert_allclose(_max_abs_error(y_true, y_pred), 0.5)


def test_benchmark_surrogate_basic():
    """Test benchmark_surrogate with a simple MLP model."""
    model = MLP(input_dim=3, output_dim=6, hidden_dims=[8])
    inputs = np.random.default_rng(42).standard_normal((50, 3)).astype(np.float32)
    targets = np.random.default_rng(42).standard_normal((50, 6)).astype(np.float32)

    result = benchmark_surrogate(
        model=model,
        test_inputs=inputs,
        test_targets=targets,
        material_name="TestMat",
        model_name="TestModel",
    )
    assert result.material_name == "TestMat"
    assert result.model_name == "TestModel"
    assert "r2" in result.metrics
    assert "rmse" in result.metrics
    assert "mae" in result.metrics
    assert "max_abs_error" in result.metrics
    assert result.n_parameters > 0


def test_benchmark_surrogate_selected_metrics():
    """Test benchmark_surrogate with a subset of metrics."""
    model = MLP(input_dim=2, output_dim=1, hidden_dims=[4])
    inputs = np.random.default_rng(0).standard_normal((20, 2)).astype(np.float32)
    targets = np.random.default_rng(0).standard_normal((20, 1)).astype(np.float32)

    result = benchmark_surrogate(
        model=model,
        test_inputs=inputs,
        test_targets=targets,
        metric_names=["rmse"],
    )
    assert "rmse" in result.metrics
    assert "r2" not in result.metrics


def test_benchmark_surrogate_with_normalizers():
    """Test benchmark_surrogate with input/output normalizers."""
    from hyper_surrogate.data.dataset import Normalizer

    rng = np.random.default_rng(42)
    raw_inputs = rng.standard_normal((50, 3)).astype(np.float32) * 10
    raw_targets = rng.standard_normal((50, 6)).astype(np.float32) * 5

    in_norm = Normalizer()
    in_norm.fit(raw_inputs)
    out_norm = Normalizer()
    out_norm.fit(raw_targets)

    model = MLP(input_dim=3, output_dim=6, hidden_dims=[8])
    result = benchmark_surrogate(
        model=model,
        test_inputs=raw_inputs,
        test_targets=raw_targets,
        input_normalizer=in_norm,
        output_normalizer=out_norm,
    )
    assert "r2" in result.metrics


@pytest.mark.slow
def test_benchmark_suite():
    """Test full benchmark suite with training."""
    from hyper_surrogate.mechanics.materials import NeoHooke
    from hyper_surrogate.training.losses import StressLoss

    materials = [NeoHooke({"C10": 0.5, "KBULK": 1000.0})]
    model_configs = [
        {
            "model_cls": MLP,
            "kwargs": {"hidden_dims": [8]},
            "loss_cls": StressLoss,
            "loss_kwargs": {},
            "epochs": 2,
            "name": "TinyMLP",
        },
    ]

    results = benchmark_suite(materials, model_configs, n_samples=100, seed=42)
    assert len(results) == 1
    assert results[0].material_name == "NeoHooke"
    assert results[0].model_name == "TinyMLP"
    assert results[0].training_time > 0
    assert "r2" in results[0].metrics


@pytest.mark.slow
def test_benchmark_suite_default_loss():
    """Test benchmark_suite with default loss (no loss_cls specified)."""
    from hyper_surrogate.mechanics.materials import NeoHooke

    materials = [NeoHooke({"C10": 0.5, "KBULK": 1000.0})]
    model_configs = [
        {
            "model_cls": MLP,
            "kwargs": {"hidden_dims": [8]},
            "epochs": 2,
        },
    ]

    results = benchmark_suite(materials, model_configs, n_samples=100, seed=42)
    assert len(results) == 1
