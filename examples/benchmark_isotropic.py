"""Benchmark surrogate models on isotropic hyperelastic materials.

Compares MLP and ICNN architectures across NeoHooke, MooneyRivlin, and Yeoh
materials using the benchmarking suite.
"""

from __future__ import annotations

from hyper_surrogate.benchmarking.metrics import benchmark_suite
from hyper_surrogate.benchmarking.reporting import results_to_markdown
from hyper_surrogate.mechanics.materials import MooneyRivlin, NeoHooke, Yeoh
from hyper_surrogate.models.icnn import ICNN
from hyper_surrogate.models.mlp import MLP
from hyper_surrogate.training.losses import StressLoss


def main() -> None:
    materials = [NeoHooke(), MooneyRivlin(), Yeoh()]

    model_configs = [
        {
            "name": "MLP-64x64",
            "model_cls": MLP,
            "kwargs": {"hidden_dims": [64, 64], "activation": "tanh"},
            "loss_cls": StressLoss,
            "epochs": 200,
            "target_type": "pk2_voigt",
        },
        {
            "name": "ICNN-64x64",
            "model_cls": ICNN,
            "kwargs": {"hidden_dims": [64, 64]},
            "loss_cls": StressLoss,
            "epochs": 200,
            "target_type": "pk2_voigt",
        },
    ]

    print("Running benchmark suite...")
    results = benchmark_suite(materials, model_configs, n_samples=5000, seed=42)

    print("\nResults:")
    print(results_to_markdown(results))

    for r in results:
        print(r.summary())
        print()


if __name__ == "__main__":
    main()
