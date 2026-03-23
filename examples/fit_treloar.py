"""Fit hyperelastic models to Treloar (1944) rubber uniaxial data.

Demonstrates parameter fitting for NeoHooke, MooneyRivlin, and Yeoh models
against the classic Treloar rubber benchmark dataset.
"""

from __future__ import annotations

from hyper_surrogate.data.experimental import ExperimentalData
from hyper_surrogate.data.fitting import fit_material
from hyper_surrogate.mechanics.materials import MooneyRivlin, NeoHooke, Yeoh


def main() -> None:
    # Load Treloar reference data
    data = ExperimentalData.load_reference("treloar")
    print(f"Loaded Treloar data: {len(data.stretch)} data points")
    print(f"Stretch range: [{data.stretch.min():.2f}, {data.stretch.max():.2f}]")
    print(f"Stress range: [{data.stress.min():.3f}, {data.stress.max():.3f}] MPa")
    print()

    # Fit NeoHooke
    mat_nh, res_nh = fit_material(
        NeoHooke,
        data,
        initial_guess={"C10": 0.1},
        fixed_params={"KBULK": 1000.0},
    )
    print(f"NeoHooke: C10={res_nh.parameters["C10"]:.4f}, R²={res_nh.r_squared:.4f}")

    # Fit MooneyRivlin
    mat_mr, res_mr = fit_material(
        MooneyRivlin,
        data,
        initial_guess={"C10": 0.1, "C01": 0.05},
        fixed_params={"KBULK": 1000.0},
    )
    print(
        f"MooneyRivlin: C10={res_mr.parameters["C10"]:.4f}, C01={res_mr.parameters["C01"]:.4f}, R²={res_mr.r_squared:.4f}"
    )

    # Fit Yeoh
    mat_ye, res_ye = fit_material(
        Yeoh,
        data,
        initial_guess={"C10": 0.1, "C20": 0.001, "C30": 0.0001},
        fixed_params={"KBULK": 1000.0},
    )
    print(
        f"Yeoh: C10={res_ye.parameters["C10"]:.4f}, C20={res_ye.parameters["C20"]:.6f}, C30={res_ye.parameters["C30"]:.8f}, R²={res_ye.r_squared:.4f}"
    )


if __name__ == "__main__":
    main()
