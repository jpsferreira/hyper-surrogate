"""Tests for benchmarking metrics and reporting."""

import numpy as np

from hyper_surrogate.benchmarking.metrics import BenchmarkResult, _mae, _r_squared, _rmse
from hyper_surrogate.benchmarking.reporting import results_to_latex, results_to_markdown


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
