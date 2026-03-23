"""Benchmark metrics for evaluating surrogate model accuracy."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class BenchmarkResult:
    """Result of benchmarking a surrogate model against analytical material."""

    material_name: str
    model_name: str
    metrics: dict[str, float] = field(default_factory=dict)
    training_time: float = 0.0
    n_parameters: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [f"Benchmark: {self.model_name} on {self.material_name}"]
        lines.append(f"  Parameters: {self.n_parameters}")
        if self.training_time > 0:
            lines.append(f"  Training time: {self.training_time:.2f}s")
        for k, v in sorted(self.metrics.items()):
            lines.append(f"  {k}: {v:.6f}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Flat dictionary for tabular export."""
        d: dict[str, Any] = {
            "material": self.material_name,
            "model": self.model_name,
            "n_parameters": self.n_parameters,
            "training_time": self.training_time,
        }
        d.update(self.metrics)
        return d


def _r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _max_abs_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.max(np.abs(y_true - y_pred)))


METRIC_FUNCS = {
    "r2": _r_squared,
    "rmse": _rmse,
    "mae": _mae,
    "max_abs_error": _max_abs_error,
}


def benchmark_surrogate(
    model: Any,
    test_inputs: np.ndarray,
    test_targets: np.ndarray,
    material_name: str = "",
    model_name: str = "",
    metric_names: list[str] | None = None,
    input_normalizer: Any | None = None,
    output_normalizer: Any | None = None,
) -> BenchmarkResult:
    """Evaluate a trained surrogate model against ground-truth targets.

    Args:
        model: Trained SurrogateModel (torch nn.Module).
        test_inputs: Raw input data (N, input_dim).
        test_targets: Ground-truth target data (N, output_dim).
        material_name: Name for reporting.
        model_name: Name for reporting.
        metric_names: Which metrics to compute. Defaults to all.
        input_normalizer: Optional Normalizer for inputs.
        output_normalizer: Optional Normalizer for targets.

    Returns:
        BenchmarkResult with computed metrics.
    """
    try:
        import torch
    except ImportError:
        msg = "torch is required for benchmarking surrogate models"
        raise ImportError(msg) from None

    if metric_names is None:
        metric_names = list(METRIC_FUNCS.keys())

    # Normalize inputs if needed
    inputs = input_normalizer.transform(test_inputs) if input_normalizer else test_inputs
    inputs_t = torch.tensor(inputs, dtype=torch.float32)

    # Predict
    model.eval()
    with torch.no_grad():
        pred_t = model(inputs_t)
    pred = pred_t.numpy()

    # Denormalize if needed
    if output_normalizer:
        pred = output_normalizer.inverse_transform(pred)

    # Compute metrics
    metrics: dict[str, float] = {}
    for name in metric_names:
        if name in METRIC_FUNCS:
            metrics[name] = METRIC_FUNCS[name](test_targets, pred)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())

    return BenchmarkResult(
        material_name=material_name,
        model_name=model_name,
        metrics=metrics,
        n_parameters=n_params,
    )


def benchmark_suite(
    materials: list[Any],
    model_configs: list[dict[str, Any]],
    n_samples: int = 10000,
    seed: int = 42,
) -> list[BenchmarkResult]:
    """Run a full benchmarking suite across materials and model configurations.

    Args:
        materials: List of Material instances to benchmark.
        model_configs: List of dicts with keys 'model_cls', 'kwargs', 'loss_cls', 'loss_kwargs', 'epochs'.
        n_samples: Number of training samples.
        seed: Random seed.

    Returns:
        List of BenchmarkResult for each (material, model) combination.
    """
    from hyper_surrogate.data.dataset import create_datasets
    from hyper_surrogate.training.trainer import Trainer

    results = []

    for material in materials:
        mat_name = type(material).__name__

        for config in model_configs:
            model_cls = config["model_cls"]
            model_kwargs = config.get("kwargs", {})
            loss_cls = config.get("loss_cls")
            loss_kwargs = config.get("loss_kwargs", {})
            epochs = config.get("epochs", 100)
            model_name = config.get("name", model_cls.__name__)

            # Create datasets
            target_type = config.get("target_type", "pk2_voigt")
            input_type = config.get("input_type", "invariants")
            train_ds, val_ds, _in_norm, _out_norm = create_datasets(
                material, n_samples, input_type=input_type, target_type=target_type, seed=seed
            )

            # Create model
            input_dim = train_ds.inputs.shape[1]
            if "input_dim" not in model_kwargs:
                model_kwargs["input_dim"] = input_dim
            if target_type == "pk2_voigt" and "output_dim" not in model_kwargs:
                model_kwargs["output_dim"] = 6

            model = model_cls(**model_kwargs)

            # Create loss
            if loss_cls is not None:
                loss_fn = loss_cls(**loss_kwargs)
            else:
                from hyper_surrogate.training.losses import StressLoss

                loss_fn = StressLoss()

            # Train
            trainer = Trainer(model, train_ds, val_ds, loss_fn=loss_fn, max_epochs=epochs)
            t0 = time.time()
            trainer.fit()
            training_time = time.time() - t0

            # Evaluate on validation set
            result = benchmark_surrogate(
                model=model,
                test_inputs=val_ds.inputs,
                test_targets=val_ds.targets,
                material_name=mat_name,
                model_name=model_name,
            )
            result.training_time = training_time

            results.append(result)

    return results
