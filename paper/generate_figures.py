"""Generate publication-quality figures from benchmark results.

Reads JSON files from results/ and produces PDF/SVG figures in figures/.

Figures:
    1. accuracy_comparison.pdf   — R² bar chart: materials × architectures
    2. convergence_curves.pdf    — Training/validation loss curves
    3. stress_scatter.pdf        — Predicted vs analytical stress (per-component)
    4. error_distribution.pdf    — Error histograms per architecture
    5. scaling_samples.pdf       — Accuracy vs training set size
    6. scaling_width.pdf         — Accuracy vs network width
    7. timing_bar.pdf            — Inference speed comparison

Usage:
    uv run python paper/generate_figures.py
    uv run python paper/generate_figures.py --format svg
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "text.usetex": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

# Color palette (colorblind-friendly)
COLORS = {
    "MLP": "#0072B2",
    "ICNN": "#D55E00",
    "Polyconvex": "#009E73",
    "CANN": "#CC79A7",
    "MLP-energy": "#56B4E9",
    "ICNN-energy": "#E69F00",
}

MARKERS = {"MLP": "o", "ICNN": "s", "Polyconvex": "D", "CANN": "^"}


def _load_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def _color_for(model_name: str) -> str:
    for key, color in COLORS.items():
        if key.lower() in model_name.lower():
            return color
    return "#333333"


def _marker_for(model_name: str) -> str:
    for key, marker in MARKERS.items():
        if key.lower() in model_name.lower():
            return marker
    return "o"


def _save(fig: plt.Figure, name: str, fmt: str) -> None:
    path = FIGURES / f"{name}.{fmt}"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ── 1. Accuracy comparison bar chart ──────────────────────────────
def plot_accuracy_comparison(fmt: str) -> None:
    data = _load_json(RESULTS / "benchmarks.json")
    if not data:
        print("  Skipping accuracy_comparison (no data)")
        return

    materials = sorted(set(d["material"] for d in data))
    models = sorted(set(d["model"] for d in data))

    # Plot R^2 for both energy and stress gradients
    for metric_key, ylabel, suffix in [
        ("energy_r2", "R² (energy)", "energy"),
        ("stress_r2", "R² (dW/dI)", "stress"),
    ]:
        fig, ax = plt.subplots(figsize=(max(6, len(materials) * 1.5), 4))
        x = np.arange(len(materials))
        width = 0.8 / max(len(models), 1)

        for i, model_name in enumerate(models):
            r2_vals = []
            for mat in materials:
                match = [d for d in data if d["material"] == mat and d["model"] == model_name]
                val = match[0].get(metric_key, 0) if match else 0
                r2_vals.append(max(val, 0))  # clip negative R^2 for display
            offset = (i - len(models) / 2 + 0.5) * width
            ax.bar(x + offset, r2_vals, width * 0.9, label=model_name, color=_color_for(model_name), edgecolor="white")

        ax.set_xlabel("Material")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Surrogate accuracy — {ylabel}")
        ax.set_xticks(x)
        ax.set_xticklabels(materials, rotation=30, ha="right")
        ax.set_ylim(0, 1.05)
        ax.legend(loc="lower left", ncol=2)
        fig.tight_layout()
        _save(fig, f"accuracy_comparison_{suffix}", fmt)


# ── 2. Convergence curves ─────────────────────────────────────────
def plot_convergence_curves(fmt: str) -> None:
    conv_files = sorted(RESULTS.glob("convergence_*.json"))
    if not conv_files:
        print("  Skipping convergence_curves (no data)")
        return

    # Group by material
    from collections import defaultdict

    by_material: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for f in conv_files:
        parts = f.stem.replace("convergence_", "").split("_", 1)
        mat_name = parts[0]
        model_name = parts[1] if len(parts) > 1 else "unknown"
        by_material[mat_name].append((model_name, _load_json(f)))

    # Plot up to 6 materials in a grid
    materials = sorted(by_material.keys())[:6]
    n_mats = len(materials)
    ncols = min(3, n_mats)
    nrows = (n_mats + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows), squeeze=False)

    for idx, mat_name in enumerate(materials):
        ax = axes[idx // ncols][idx % ncols]
        for model_name, history in by_material[mat_name]:
            val_loss = history.get("val_loss", [])
            if val_loss:
                ax.semilogy(val_loss, label=model_name, color=_color_for(model_name), alpha=0.8)
        ax.set_title(mat_name)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation loss")
        ax.legend(fontsize=7, loc="upper right")

    # Hide empty subplots
    for idx in range(n_mats, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.tight_layout()
    _save(fig, "convergence_curves", fmt)


# ── 3. Predicted vs analytical scatter ─────────────────────────────
def plot_stress_scatter(fmt: str) -> None:
    """Train models on NeoHooke, scatter predicted vs true dW/dI (stress gradients)."""
    try:
        import torch

        from hyper_surrogate.data.dataset import MaterialDataset, Normalizer
        from hyper_surrogate.data.deformation import DeformationGenerator
        from hyper_surrogate.mechanics.kinematics import Kinematics
        from hyper_surrogate.mechanics.materials import NeoHooke
        from hyper_surrogate.models.icnn import ICNN
        from hyper_surrogate.models.mlp import MLP
        from hyper_surrogate.training.losses import EnergyStressLoss
        from hyper_surrogate.training.trainer import Trainer
    except ImportError:
        print("  Skipping stress_scatter (torch not available)")
        return

    material = NeoHooke()
    gen = DeformationGenerator(seed=42)
    F = gen.combined(5000, stretch_range=(0.7, 1.5), shear_range=(-0.3, 0.3))
    C = Kinematics.right_cauchy_green(F)
    i1 = Kinematics.isochoric_invariant1(C)
    i2 = Kinematics.isochoric_invariant2(C)
    j = np.sqrt(Kinematics.det_invariant(C))
    inputs = np.column_stack([i1, i2, j])
    energy = material.evaluate_energy(C)
    dW_dI = material.evaluate_energy_grad_invariants(C)

    in_norm = Normalizer().fit(inputs)
    X = in_norm.transform(inputs).astype(np.float32)
    W = energy.reshape(-1, 1).astype(np.float32)
    S = (dW_dI * in_norm.params["std"]).astype(np.float32)

    n_val = int(5000 * 0.15)
    idx = np.random.default_rng(42).permutation(5000)
    ti, vi = idx[n_val:], idx[:n_val]
    train_ds = MaterialDataset(X[ti], (W[ti], S[ti]))
    val_ds = MaterialDataset(X[vi], (W[vi], S[vi]))
    raw_dW_dI_val = dW_dI[vi]

    models_to_plot = [
        ("MLP", MLP(input_dim=3, output_dim=1, hidden_dims=[64, 64], activation="softplus")),
        ("ICNN", ICNN(input_dim=3, hidden_dims=[64, 64])),
    ]

    fig, axes = plt.subplots(1, len(models_to_plot), figsize=(5 * len(models_to_plot), 4.5))
    if len(models_to_plot) == 1:
        axes = [axes]

    grad_labels = ["dW/dI1", "dW/dI2", "dW/dJ"]

    for ax, (name, model) in zip(axes, models_to_plot):
        Trainer(
            model, train_ds, val_ds,
            loss_fn=EnergyStressLoss(alpha=1.0, beta=1.0),
            max_epochs=500, patience=100,
        ).fit()
        model.eval()

        x_t = torch.tensor(val_ds.inputs, dtype=torch.float32, requires_grad=True)
        pred_W = model(x_t)
        dW_dx = torch.autograd.grad(pred_W.sum(), x_t)[0]
        pred_dW_dI = dW_dx.detach().numpy() / in_norm.params["std"]

        for c in range(3):
            ax.scatter(raw_dW_dI_val[:, c], pred_dW_dI[:, c], s=3, alpha=0.3, label=grad_labels[c])

        lims = [
            min(raw_dW_dI_val.min(), pred_dW_dI.min()),
            max(raw_dW_dI_val.max(), pred_dW_dI.max()),
        ]
        ax.plot(lims, lims, "k--", lw=0.8, alpha=0.5)
        ax.set_xlabel("Analytical dW/dI")
        ax.set_ylabel("Predicted dW/dI")
        ax.set_title(f"{name} — NeoHooke")
        ax.legend(fontsize=7, markerscale=3, loc="upper left")
        ax.set_aspect("equal")

    fig.tight_layout()
    _save(fig, "stress_scatter", fmt)


# ── 4. Error distribution ─────────────────────────────────────────
def plot_error_distribution(fmt: str) -> None:
    """Histogram of relative errors for each architecture (energy-based dW/dI)."""
    try:
        import torch

        from hyper_surrogate.data.dataset import MaterialDataset, Normalizer
        from hyper_surrogate.data.deformation import DeformationGenerator
        from hyper_surrogate.mechanics.kinematics import Kinematics
        from hyper_surrogate.mechanics.materials import NeoHooke
        from hyper_surrogate.models.icnn import ICNN
        from hyper_surrogate.models.mlp import MLP
        from hyper_surrogate.training.losses import EnergyStressLoss
        from hyper_surrogate.training.trainer import Trainer
    except ImportError:
        print("  Skipping error_distribution (torch not available)")
        return

    material = NeoHooke()
    gen = DeformationGenerator(seed=42)
    F = gen.combined(5000, stretch_range=(0.7, 1.5), shear_range=(-0.3, 0.3))
    C = Kinematics.right_cauchy_green(F)
    i1 = Kinematics.isochoric_invariant1(C)
    i2 = Kinematics.isochoric_invariant2(C)
    j = np.sqrt(Kinematics.det_invariant(C))
    inputs = np.column_stack([i1, i2, j])
    energy = material.evaluate_energy(C)
    dW_dI = material.evaluate_energy_grad_invariants(C)

    in_norm = Normalizer().fit(inputs)
    X = in_norm.transform(inputs).astype(np.float32)
    W = energy.reshape(-1, 1).astype(np.float32)
    S = (dW_dI * in_norm.params["std"]).astype(np.float32)

    n_val = int(5000 * 0.15)
    idx = np.random.default_rng(42).permutation(5000)
    ti, vi = idx[n_val:], idx[:n_val]
    train_ds = MaterialDataset(X[ti], (W[ti], S[ti]))
    val_ds = MaterialDataset(X[vi], (W[vi], S[vi]))
    raw_dW_dI_val = dW_dI[vi]

    architectures = [
        ("MLP", MLP(input_dim=3, output_dim=1, hidden_dims=[64, 64], activation="softplus")),
        ("ICNN", ICNN(input_dim=3, hidden_dims=[64, 64])),
    ]

    fig, ax = plt.subplots(figsize=(6, 4))

    for name, model in architectures:
        Trainer(
            model, train_ds, val_ds,
            loss_fn=EnergyStressLoss(alpha=1.0, beta=1.0),
            max_epochs=500, patience=100,
        ).fit()
        model.eval()

        x_t = torch.tensor(val_ds.inputs, dtype=torch.float32, requires_grad=True)
        pred_W = model(x_t)
        dW_dx = torch.autograd.grad(pred_W.sum(), x_t)[0]
        pred_dW_dI = dW_dx.detach().numpy() / in_norm.params["std"]

        denom = np.abs(raw_dW_dI_val)
        denom[denom < 1e-10] = 1e-10
        rel_error = np.abs(pred_dW_dI - raw_dW_dI_val) / denom
        rel_error_flat = rel_error.flatten()
        rel_error_flat = rel_error_flat[rel_error_flat < 1.0]

        ax.hist(
            rel_error_flat, bins=100, alpha=0.6, density=True,
            label=f"{name} (median={np.median(rel_error_flat):.2e})",
            color=_color_for(name),
        )

    ax.set_xlabel("Relative error")
    ax.set_ylabel("Density")
    ax.set_title("Error distribution — NeoHooke dW/dI")
    ax.legend()
    ax.set_xlim(0, 0.2)
    fig.tight_layout()
    _save(fig, "error_distribution", fmt)


# ── 5. Scaling study plots ────────────────────────────────────────
def plot_scaling(fmt: str) -> None:
    path = RESULTS / "scaling_study.json"
    if not path.exists():
        print("  Skipping scaling plots (no data)")
        return

    data = _load_json(path)
    sample_data = [d for d in data if d["study"] == "sample_size"]
    width_data = [d for d in data if d["study"] == "network_width"]

    if sample_data:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ns = [d["n_samples"] for d in sample_data]
        r2s = [d.get("stress_r2", d.get("r2", 0)) for d in sample_data]
        ax.semilogx(ns, r2s, "o-", color=COLORS["MLP"], markersize=6)
        ax.set_xlabel("Training samples")
        ax.set_ylabel("R²")
        ax.set_title("Accuracy vs training set size (MLP-64x64, NeoHooke)")
        fig.tight_layout()
        _save(fig, "scaling_samples", fmt)

    if width_data:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ws = [d["width"] for d in width_data]
        r2s = [d.get("stress_r2", d.get("r2", 0)) for d in width_data]
        n_params = [d.get("n_parameters", 0) for d in width_data]
        ax.plot(ws, r2s, "s-", color=COLORS["MLP"], markersize=6)
        ax.set_xlabel("Hidden layer width")
        ax.set_ylabel("R²")
        ax.set_title("Accuracy vs network width (NeoHooke)")

        ax2 = ax.twinx()
        ax2.bar(ws, n_params, width=[w * 0.3 for w in ws], alpha=0.2, color=COLORS["MLP"])
        ax2.set_ylabel("Parameters", color=COLORS["MLP"], alpha=0.5)

        fig.tight_layout()
        _save(fig, "scaling_width", fmt)


# ── 6. Timing bar chart ──────────────────────────────────────────
def plot_timing(fmt: str) -> None:
    path = RESULTS / "timing.json"
    if not path.exists():
        print("  Skipping timing plot (no data)")
        return

    data = _load_json(path)

    fig, ax = plt.subplots(figsize=(4, 3.5))
    methods = ["Analytical", "NN (MLP)"]
    means = [data["analytical_mean_ms"], data["nn_mean_ms"]]
    stds = [data["analytical_std_ms"], data["nn_std_ms"]]
    colors = ["#999999", COLORS["MLP"]]

    bars = ax.bar(methods, means, yerr=stds, capsize=5, color=colors, edgecolor="white", width=0.5)
    ax.set_ylabel("Inference time (ms)")
    ax.set_title(f"Evaluation speed ({data['n_eval']} samples)")

    # Annotate speedup
    if data["speedup"] > 1:
        ax.annotate(
            f"{data['speedup']:.1f}× faster",
            xy=(1, means[1]), xytext=(1.3, means[0] * 0.5),
            arrowprops=dict(arrowstyle="->", color="black"),
            fontsize=10, fontweight="bold",
        )

    fig.tight_layout()
    _save(fig, "timing_bar", fmt)


# ── main ───────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--format", default="pdf", choices=["pdf", "svg", "png"], help="Output format")
    args = parser.parse_args()
    fmt = args.format

    print(f"Generating figures (format={fmt})")

    plot_accuracy_comparison(fmt)
    plot_convergence_curves(fmt)
    plot_stress_scatter(fmt)
    plot_error_distribution(fmt)
    plot_scaling(fmt)
    plot_timing(fmt)

    print(f"\nAll figures saved to {FIGURES}/")


if __name__ == "__main__":
    main()
