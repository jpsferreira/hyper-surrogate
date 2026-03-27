"""Plot FE validation results: compare analytical vs surrogate UMAT.

Reads:
    results/fe_validation/validation_reference.json  — analytical reference
    results/fe_validation/abaqus_*.csv               — Abaqus output (user-provided)
    results/fe_validation/feap_*.csv                  — FEAP output (user-provided)

After running your FE simulations, extract stress-stretch data into CSV files
with columns: stretch, S11 (or sigma11). Place them in results/fe_validation/.

Expected CSV format:
    stretch,sigma11,sigma22,sigma33
    1.0,0.0,0.0,0.0
    1.01,0.123,0.001,0.001
    ...

Usage:
    uv run python paper/plot_fe_results.py
    uv run python paper/plot_fe_results.py --format svg
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

ROOT = Path(__file__).resolve().parent
FE_DIR = ROOT / "results" / "fe_validation"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

TESTS = ["uniaxial", "biaxial", "shear"]


def _load_csv(path: Path) -> dict[str, np.ndarray] | None:
    """Load a CSV with header row. Returns dict of column name -> values."""
    if not path.exists():
        return None
    data = np.genfromtxt(path, delimiter=",", names=True)
    return {name: data[name] for name in data.dtype.names}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", default="pdf", choices=["pdf", "svg", "png"])
    args = parser.parse_args()

    ref_path = FE_DIR / "validation_reference.json"
    if not ref_path.exists():
        print("No reference data found. Run `paper/fe_validation.py` first.")
        return

    with open(ref_path) as f:
        ref = json.load(f)

    mat_name = ref["material"]
    print(f"Plotting FE validation for {mat_name}")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for ax, test_name in zip(axes, TESTS):
        test_ref = ref["tests"][test_name]
        stretch = np.array(test_ref["stretch"])
        cauchy_ref = np.array(test_ref["cauchy_11"])

        # Plot analytical reference
        ax.plot(stretch, cauchy_ref, "k-", linewidth=2, label="Analytical (exact)")

        # Try loading Abaqus results
        for solver, style, label_prefix in [
            ("abaqus", "--", "Abaqus"),
            ("feap", ":", "FEAP"),
        ]:
            # Analytical UMAT results
            csv_analytical = _load_csv(FE_DIR / f"{solver}_{test_name}_analytical.csv")
            if csv_analytical is not None:
                ax.plot(
                    csv_analytical["stretch"], csv_analytical["sigma11"],
                    style, color="#0072B2", linewidth=1.5,
                    label=f"{label_prefix} analytical UMAT",
                )

            # Hybrid NN UMAT results
            csv_hybrid = _load_csv(FE_DIR / f"{solver}_{test_name}_hybrid.csv")
            if csv_hybrid is not None:
                ax.plot(
                    csv_hybrid["stretch"], csv_hybrid["sigma11"],
                    style, color="#D55E00", linewidth=1.5,
                    label=f"{label_prefix} NN UMAT",
                )

        x_label = "Stretch" if test_name != "shear" else "Shear parameter"
        ax.set_xlabel(x_label)
        ax.set_ylabel("Cauchy stress $\\sigma_{11}$")
        ax.set_title(f"{test_name.capitalize()} — {mat_name}")
        ax.legend(fontsize=8)

    fig.tight_layout()
    out = FIGURES / f"fe_validation.{args.format}"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")

    # Also plot PK2 stress
    fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4))
    for ax, test_name in zip(axes2, TESTS):
        test_ref = ref["tests"][test_name]
        stretch = np.array(test_ref["stretch"])
        pk2_11 = np.array(test_ref["pk2_11"])
        pk2_22 = np.array(test_ref["pk2_22"])

        ax.plot(stretch, pk2_11, "k-", linewidth=2, label="$S_{11}$ (analytical)")
        ax.plot(stretch, pk2_22, "k--", linewidth=2, label="$S_{22}$ (analytical)")

        x_label = "Stretch" if test_name != "shear" else "Shear parameter"
        ax.set_xlabel(x_label)
        ax.set_ylabel("PK2 stress")
        ax.set_title(f"{test_name.capitalize()} — {mat_name}")
        ax.legend(fontsize=8)

    fig2.tight_layout()
    out2 = FIGURES / f"fe_validation_pk2.{args.format}"
    fig2.savefig(out2)
    plt.close(fig2)
    print(f"  Saved {out2}")

    # Energy plot
    fig3, axes3 = plt.subplots(1, 3, figsize=(14, 4))
    for ax, test_name in zip(axes3, TESTS):
        test_ref = ref["tests"][test_name]
        stretch = np.array(test_ref["stretch"])
        energy = np.array(test_ref["energy"])

        ax.plot(stretch, energy, "k-", linewidth=2, label="$W$ (analytical)")

        x_label = "Stretch" if test_name != "shear" else "Shear parameter"
        ax.set_xlabel(x_label)
        ax.set_ylabel("Strain energy $W$")
        ax.set_title(f"{test_name.capitalize()} — {mat_name}")
        ax.legend(fontsize=8)

    fig3.tight_layout()
    out3 = FIGURES / f"fe_validation_energy.{args.format}"
    fig3.savefig(out3)
    plt.close(fig3)
    print(f"  Saved {out3}")


if __name__ == "__main__":
    main()
