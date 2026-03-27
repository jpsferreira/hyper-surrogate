"""Comprehensive benchmarks: all materials x all architectures.

Outputs:
    results/benchmarks.json          -- raw metrics
    results/benchmarks.tex           -- LaTeX table
    results/benchmarks.md            -- Markdown table
    results/convergence_{mat}_{model}.json -- per-run training curves
    results/scaling_study.json       -- accuracy vs sample size / network size
    results/timing.json              -- inference speed: NN vs analytical

Usage:
    uv run python paper/run_benchmarks.py              # full suite
    uv run python paper/run_benchmarks.py --quick      # reduced (fewer epochs/samples)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

# -- paths -----------------------------------------------------------
ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

# -- hyper-surrogate imports -----------------------------------------
from hyper_surrogate.benchmarking.metrics import BenchmarkResult
from hyper_surrogate.benchmarking.reporting import results_to_latex, results_to_markdown
from hyper_surrogate.data.dataset import MaterialDataset, Normalizer, create_datasets
from hyper_surrogate.data.deformation import DeformationGenerator
from hyper_surrogate.mechanics.kinematics import Kinematics
from hyper_surrogate.mechanics.materials import (
    GasserOgdenHolzapfel,
    HolzapfelOgden,
    MooneyRivlin,
    NeoHooke,
    Yeoh,
)
from hyper_surrogate.models.icnn import ICNN
from hyper_surrogate.models.mlp import MLP
from hyper_surrogate.models.polyconvex import PolyconvexICNN
from hyper_surrogate.training.losses import EnergyStressLoss
from hyper_surrogate.training.trainer import Trainer


# -- helpers ---------------------------------------------------------
def _save_json(obj: Any, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def _isotropic_materials() -> list[tuple[str, Any]]:
    return [
        ("NeoHooke", NeoHooke()),
        ("MooneyRivlin", MooneyRivlin()),
        ("Yeoh", Yeoh()),
    ]


def _anisotropic_materials() -> list[tuple[str, Any]]:
    a0 = np.array([1.0, 0.0, 0.0])
    # Use reduced stiffness parameters to avoid exp overflow with combined deformations.
    # These are still physiologically reasonable (soft tissue range).
    return [
        ("HolzapfelOgden", HolzapfelOgden(
            parameters={"a": 0.059, "b": 4.0, "af": 2.0, "bf": 4.0, "KBULK": 100.0},
            fiber_direction=a0,
        )),
        ("GOH", GasserOgdenHolzapfel(
            parameters={"a": 0.059, "b": 4.0, "af": 2.0, "bf": 4.0, "kappa": 0.226, "KBULK": 100.0},
            fiber_direction=a0,
        )),
    ]


def _create_anisotropic_datasets(
    material: Any,
    n_samples: int,
    seed: int = 42,
    val_fraction: float = 0.15,
) -> tuple[MaterialDataset, MaterialDataset, Normalizer, Normalizer]:
    """Build energy datasets for anisotropic materials (bypasses evaluate_energy).

    Computes W and dW/dI via sef_from_all_invariants, since the base evaluate_energy
    relies on sef which is not available for fiber-based models.
    """
    from sympy import Symbol, diff
    from sympy import lambdify as sym_lambdify

    gen = DeformationGenerator(seed=seed)
    # Tighter range for anisotropic models to avoid exp overflow in fiber terms
    F = gen.combined(n_samples, stretch_range=(0.85, 1.2), shear_range=(-0.15, 0.15))
    C = Kinematics.right_cauchy_green(F)

    # Compute invariants
    i1 = Kinematics.isochoric_invariant1(C)
    i2 = Kinematics.isochoric_invariant2(C)
    j = np.sqrt(Kinematics.det_invariant(C))

    i1s, i2s, js = Symbol("I1b"), Symbol("I2b"), Symbol("J")
    inv_syms: list[Symbol] = [i1s, i2s, js]
    inv_vals: list[np.ndarray] = [i1, i2, j]

    fiber_inv_pairs: list[tuple[Symbol, Symbol]] = []
    for k, a0 in enumerate(material._fiber_directions):
        i4k = Symbol(f"I4_{k}")
        i5k = Symbol(f"I5_{k}")
        inv_syms.extend([i4k, i5k])
        fiber_inv_pairs.append((i4k, i5k))
        inv_vals.append(Kinematics.fiber_invariant4(C, a0))
        inv_vals.append(Kinematics.fiber_invariant5(C, a0))

    # Build symbolic W and dW/dI
    W_sym = material.sef_from_all_invariants(i1s, i2s, js, fiber_inv_pairs or None)
    dW_syms = [diff(W_sym, s) for s in inv_syms]

    param_syms = list(material._symbols.values())
    param_vals = list(material._params.values())

    # Lambdify energy
    W_fn = sym_lambdify((*inv_syms, *param_syms), W_sym, modules="numpy")
    dW_fn = sym_lambdify((*inv_syms, *param_syms), dW_syms, modules="numpy")

    energy = np.array(W_fn(*inv_vals, *param_vals), dtype=float).flatten()
    dW_raw = dW_fn(*inv_vals, *param_vals)
    n = len(C)
    dW_dI = np.column_stack([np.broadcast_to(np.asarray(r, dtype=float), (n,)) for r in dW_raw])

    # Build inputs
    inputs = np.column_stack(inv_vals)

    # Normalize and build datasets (same as create_datasets energy path)
    in_norm = Normalizer().fit(inputs)
    X = in_norm.transform(inputs).astype(np.float32)
    W = energy.reshape(-1, 1).astype(np.float32)
    S = (dW_dI * in_norm.params["std"]).astype(np.float32)

    n_val = int(n * val_fraction)
    idx = np.random.default_rng(seed).permutation(n)
    ti, vi = idx[n_val:], idx[:n_val]

    train_ds = MaterialDataset(X[ti], (W[ti], S[ti]))
    val_ds = MaterialDataset(X[vi], (W[vi], S[vi]))
    energy_norm = Normalizer().fit(energy.reshape(-1, 1))

    return train_ds, val_ds, in_norm, energy_norm


def _train_and_eval_from_datasets(
    mat_name: str,
    model_name: str,
    model: torch.nn.Module,
    train_ds: MaterialDataset,
    val_ds: MaterialDataset,
    epochs: int,
    patience: int,
) -> tuple[BenchmarkResult, dict[str, list[float]]]:
    """Train and evaluate from pre-built datasets."""
    trainer = Trainer(
        model, train_ds, val_ds,
        loss_fn=EnergyStressLoss(alpha=1.0, beta=1.0),
        max_epochs=epochs, lr=1e-3, patience=patience, batch_size=256,
    )
    t0 = time.time()
    tr_result = trainer.fit()
    training_time = time.time() - t0

    tr_result.model.eval()
    x_t = torch.tensor(val_ds.inputs, dtype=torch.float32, requires_grad=True)
    pred_W = tr_result.model(x_t)
    dW_dx = torch.autograd.grad(pred_W.sum(), x_t, create_graph=False)[0]

    true_W = val_ds.targets[0].flatten()
    true_S = val_ds.targets[1]
    pred_W_np = pred_W.detach().numpy().flatten()
    pred_S_np = dW_dx.detach().numpy()

    from hyper_surrogate.benchmarking.metrics import METRIC_FUNCS

    metrics: dict[str, float] = {}
    for name, fn in METRIC_FUNCS.items():
        metrics[f"energy_{name}"] = fn(true_W, pred_W_np)
    for name, fn in METRIC_FUNCS.items():
        metrics[f"stress_{name}"] = fn(true_S.flatten(), pred_S_np.flatten())

    n_params = sum(p.numel() for p in tr_result.model.parameters())
    result = BenchmarkResult(
        material_name=mat_name, model_name=model_name,
        metrics=metrics, n_parameters=n_params, training_time=training_time,
    )
    return result, tr_result.history


def _train_and_eval(
    material: Any,
    mat_name: str,
    model_name: str,
    model: torch.nn.Module,
    n_samples: int,
    epochs: int,
    patience: int,
    seed: int = 42,
) -> tuple[BenchmarkResult, dict[str, list[float]]]:
    """Train energy-based model using create_datasets and evaluate accuracy.

    Uses the library's create_datasets(target_type="energy") which handles
    normalization correctly for the EnergyStressLoss.
    """
    train_ds, val_ds, in_norm, energy_norm = create_datasets(
        material, n_samples, input_type="invariants", target_type="energy", seed=seed
    )

    trainer = Trainer(
        model, train_ds, val_ds,
        loss_fn=EnergyStressLoss(alpha=1.0, beta=1.0),
        max_epochs=epochs, lr=1e-3, patience=patience, batch_size=256,
    )

    t0 = time.time()
    tr_result = trainer.fit()
    training_time = time.time() - t0

    # Evaluate: predict energy and compute dW/dI via autograd
    tr_result.model.eval()
    x_t = torch.tensor(val_ds.inputs, dtype=torch.float32, requires_grad=True)
    pred_W_norm = tr_result.model(x_t)
    dW_dx = torch.autograd.grad(pred_W_norm.sum(), x_t, create_graph=False)[0]

    pred_W_norm_np = pred_W_norm.detach().numpy().flatten()
    dW_dx_np = dW_dx.detach().numpy()

    # val_ds.targets = (W_raw [N,1], S_scaled [N, input_dim])
    # where S_scaled = dW/dI * in_std. Energy is raw (not normalized).
    true_W = val_ds.targets[0].flatten()  # (N,)
    true_S = val_ds.targets[1]  # (N, input_dim) - these are dW/dI * std

    # Predicted energy (model outputs raw W since targets are raw)
    pred_W = pred_W_norm_np
    # Predicted scaled stress gradients: dW/d(x_norm) directly comparable to true_S
    pred_dW_dI = dW_dx_np

    # Compute metrics
    from hyper_surrogate.benchmarking.metrics import METRIC_FUNCS

    metrics: dict[str, float] = {}
    for name, fn in METRIC_FUNCS.items():
        metrics[f"energy_{name}"] = fn(true_W, pred_W)
    for name, fn in METRIC_FUNCS.items():
        metrics[f"stress_{name}"] = fn(true_S.flatten(), pred_dW_dI.flatten())

    n_params = sum(p.numel() for p in tr_result.model.parameters())

    result = BenchmarkResult(
        material_name=mat_name,
        model_name=model_name,
        metrics=metrics,
        n_parameters=n_params,
        training_time=training_time,
    )

    return result, tr_result.history


# -- 1. Main accuracy benchmarks ------------------------------------
def run_accuracy_benchmarks(quick: bool) -> list[dict[str, Any]]:
    """Run all material x architecture combinations (energy-based training)."""
    n_samples = 3000 if quick else 10000
    epochs = 300 if quick else 1000
    patience = 60 if quick else 200
    hidden = [32, 32] if quick else [64, 64]
    h_str = "x".join(map(str, hidden))

    all_results: list[dict[str, Any]] = []
    bench_results: list[BenchmarkResult] = []

    for mat_name, material in _isotropic_materials():
        print(f"\n== {mat_name} ==")

        model_configs: list[tuple[str, torch.nn.Module]] = [
            (f"MLP-{h_str}", MLP(input_dim=3, output_dim=1, hidden_dims=hidden, activation="softplus")),
            (f"ICNN-{h_str}", ICNN(input_dim=3, hidden_dims=hidden)),
            (f"Polyconvex-{h_str}", PolyconvexICNN(groups=[[0], [1], [2]], hidden_dims=hidden)),
            # CANN is excluded from general benchmarks — its exponential basis functions
            # are sensitive to input scale and require material-specific tuning.
            # Use examples/model_discovery_cann.py for CANN-specific workflows.
        ]

        for model_name, model in model_configs:
            print(f"  {model_name} ...", end=" ", flush=True)
            try:
                result, history = _train_and_eval(
                    material, mat_name, model_name, model, n_samples, epochs, patience
                )
                r2_e = result.metrics.get("energy_r2", 0)
                r2_s = result.metrics.get("stress_r2", 0)
                print(f"R2(W)={r2_e:.4f}  R2(dW/dI)={r2_s:.4f}  time={result.training_time:.1f}s")

                bench_results.append(result)
                all_results.append(result.to_dict())
                _save_json(history, RESULTS / f"convergence_{mat_name}_{model_name.replace('/', '-')}.json")
            except Exception as e:
                print(f"FAILED: {e}")

    # --- Anisotropic materials (5D invariant input: I1, I2, J, I4, I5) ---
    for mat_name, material in _anisotropic_materials():
        print(f"\n== {mat_name} (anisotropic) ==")
        try:
            train_ds, val_ds, in_norm, energy_norm = _create_anisotropic_datasets(
                material, n_samples
            )
        except Exception as e:
            print(f"  SKIP: data generation failed ({e})")
            continue

        input_dim = train_ds.inputs.shape[1]  # 5 for single-fiber

        model_configs: list[tuple[str, torch.nn.Module]] = [
            (f"MLP-{h_str}", MLP(input_dim=input_dim, output_dim=1, hidden_dims=hidden, activation="softplus")),
            (f"ICNN-{h_str}", ICNN(input_dim=input_dim, hidden_dims=hidden)),
        ]

        for model_name, model in model_configs:
            print(f"  {model_name} ...", end=" ", flush=True)
            try:
                result, history = _train_and_eval_from_datasets(
                    mat_name, model_name, model, train_ds, val_ds, epochs, patience
                )
                r2_e = result.metrics.get("energy_r2", 0)
                r2_s = result.metrics.get("stress_r2", 0)
                print(f"R2(W)={r2_e:.4f}  R2(dW/dI)={r2_s:.4f}  time={result.training_time:.1f}s")

                bench_results.append(result)
                all_results.append(result.to_dict())
                _save_json(history, RESULTS / f"convergence_{mat_name}_{model_name.replace('/', '-')}.json")
            except Exception as e:
                print(f"FAILED: {e}")

    # Save tables
    _save_json(all_results, RESULTS / "benchmarks.json")
    (RESULTS / "benchmarks.tex").write_text(
        results_to_latex(bench_results, caption="Surrogate accuracy benchmarks")
    )
    (RESULTS / "benchmarks.md").write_text(results_to_markdown(bench_results))

    print(f"\nSaved {len(all_results)} results to {RESULTS}/benchmarks.*")
    return all_results


# -- 2. Scaling study -----------------------------------------------
def run_scaling_study(quick: bool) -> None:
    """Accuracy vs training set size and network width."""
    material = NeoHooke()
    mat_name = "NeoHooke"
    sample_sizes = [500, 1000, 3000] if quick else [500, 1000, 2000, 5000, 10000, 20000]
    widths = [16, 32, 64] if quick else [16, 32, 64, 128, 256]
    epochs = 300 if quick else 500
    patience = 60 if quick else 100

    results: list[dict[str, Any]] = []

    print("\n== Scaling study: sample size ==")
    for n in sample_sizes:
        print(f"  n={n} ...", end=" ", flush=True)
        model = MLP(input_dim=3, output_dim=1, hidden_dims=[64, 64], activation="softplus")
        result, _ = _train_and_eval(material, mat_name, "MLP-64x64", model, n, epochs, patience)
        r2 = result.metrics.get("stress_r2", 0)
        print(f"R2(dW/dI)={r2:.4f}")
        results.append({"study": "sample_size", "n_samples": n, "width": 64, **result.to_dict()})

    print("\n== Scaling study: network width ==")
    n_fixed = 3000 if quick else 5000
    for w in widths:
        print(f"  width={w} ...", end=" ", flush=True)
        model = MLP(input_dim=3, output_dim=1, hidden_dims=[w, w], activation="softplus")
        result, _ = _train_and_eval(material, mat_name, f"MLP-{w}x{w}", model, n_fixed, epochs, patience)
        r2 = result.metrics.get("stress_r2", 0)
        print(f"R2(dW/dI)={r2:.4f}  params={result.n_parameters}")
        results.append({"study": "network_width", "n_samples": n_fixed, "width": w, **result.to_dict()})

    _save_json(results, RESULTS / "scaling_study.json")
    print(f"  Saved to {RESULTS}/scaling_study.json")


# -- 3. Inference timing ---------------------------------------------
def run_timing_benchmarks(quick: bool) -> None:
    """Compare NN inference speed vs analytical material evaluation."""
    n_eval = 1000 if quick else 10000
    n_repeats = 5 if quick else 20
    material = NeoHooke()

    gen = DeformationGenerator(seed=0)
    F = gen.combined(n_eval, stretch_range=(0.7, 1.5), shear_range=(-0.3, 0.3))
    C = Kinematics.right_cauchy_green(F)
    i1 = Kinematics.isochoric_invariant1(C)
    i2 = Kinematics.isochoric_invariant2(C)
    j = np.sqrt(Kinematics.det_invariant(C))
    inputs_np = np.column_stack([i1, i2, j]).astype(np.float32)
    inputs_t = torch.tensor(inputs_np)

    # Analytical timing
    times_analytical: list[float] = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        _ = material.evaluate_pk2(C)
        times_analytical.append(time.perf_counter() - t0)

    # Train a small MLP for timing
    model = MLP(input_dim=3, output_dim=1, hidden_dims=[64, 64], activation="softplus")
    train_ds, val_ds, in_norm, _ = create_datasets(material, 2000, target_type="energy", seed=42)
    Trainer(model, train_ds, val_ds, loss_fn=EnergyStressLoss(), max_epochs=100, patience=30).fit()
    model.eval()

    # NN timing (normalize + forward)
    in_mean = torch.tensor(in_norm.params["mean"], dtype=torch.float32)
    in_std = torch.tensor(in_norm.params["std"], dtype=torch.float32)
    times_nn: list[float] = []
    with torch.no_grad():
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            x = (inputs_t - in_mean) / in_std
            _ = model(x)
            times_nn.append(time.perf_counter() - t0)

    timing = {
        "n_eval": n_eval,
        "n_repeats": n_repeats,
        "analytical_mean_ms": float(np.mean(times_analytical) * 1000),
        "analytical_std_ms": float(np.std(times_analytical) * 1000),
        "nn_mean_ms": float(np.mean(times_nn) * 1000),
        "nn_std_ms": float(np.std(times_nn) * 1000),
        "speedup": float(np.mean(times_analytical) / np.mean(times_nn)),
    }

    _save_json(timing, RESULTS / "timing.json")
    print(f"\n== Inference timing ({n_eval} samples, {n_repeats} repeats) ==")
    print(f"  Analytical: {timing['analytical_mean_ms']:.2f} +/- {timing['analytical_std_ms']:.2f} ms")
    print(f"  NN (MLP):   {timing['nn_mean_ms']:.2f} +/- {timing['nn_std_ms']:.2f} ms")
    print(f"  Speedup:    {timing['speedup']:.1f}x")


# -- main ------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Run paper benchmarks")
    parser.add_argument("--quick", action="store_true", help="Reduced run for testing")
    args = parser.parse_args()

    print(f"{'Quick' if args.quick else 'Full'} benchmark suite")
    print(f"Results directory: {RESULTS}")

    run_accuracy_benchmarks(args.quick)
    run_scaling_study(args.quick)
    run_timing_benchmarks(args.quick)

    print("\nAll benchmarks complete. Run `paper/generate_figures.py` to create plots.")


if __name__ == "__main__":
    main()
