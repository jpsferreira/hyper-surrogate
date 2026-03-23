"""Two-fiber arterial wall model using GasserOgdenHolzapfel.

Demonstrates setting up a two-fiber GOH model for arterial wall tissue,
computing energy and stress under equibiaxial loading.
"""

from __future__ import annotations

import numpy as np

from hyper_surrogate.mechanics.kinematics import Kinematics
from hyper_surrogate.mechanics.materials import GasserOgdenHolzapfel


def main() -> None:
    # Define two fiber families at +/- 39 degrees from circumferential direction
    theta = np.radians(39.0)
    fiber1 = np.array([np.cos(theta), np.sin(theta), 0.0])
    fiber2 = np.array([np.cos(theta), -np.sin(theta), 0.0])

    # Typical medial layer parameters (Holzapfel et al., 2005)
    params = {
        "a": 3.0,  # kPa (ground substance stiffness)
        "b": 0.5,
        "af": 2.3632,  # kPa (fiber stiffness)
        "bf": 0.8393,
        "kappa": 0.226,  # dispersion parameter
        "KBULK": 100.0,
    }

    # Create models for each fiber family
    mat1 = GasserOgdenHolzapfel(parameters=params, fiber_direction=fiber1)
    mat2 = GasserOgdenHolzapfel(parameters=params, fiber_direction=fiber2)

    # Generate equibiaxial stretch in 1-2 plane
    stretches = np.linspace(1.0, 1.3, 10)
    print(f"{"Stretch":>8s} {"W_fiber1":>12s} {"W_fiber2":>12s} {"W_total":>12s}")
    print("-" * 50)

    # Build symbolic energy: shared isotropic/volumetric + two fiber contributions.
    # Using a single material's SEF for the shared terms avoids double-counting.
    from sympy import Symbol, lambdify

    i1s, i2s, js = Symbol("I1b"), Symbol("I2b"), Symbol("J")
    i4s_1, i5s_1 = Symbol("I4_1"), Symbol("I5_1")
    i4s_2, i5s_2 = Symbol("I4_2"), Symbol("I5_2")

    # Full energy from fiber 1 (includes isotropic + volumetric + fiber)
    W1_expr = mat1.sef_from_invariants(i1s, i2s, js, i4s_1, i5s_1)
    # Fiber-only energy from fiber 2 (subtract isotropic + volumetric to avoid double-counting)
    W2_full = mat2.sef_from_invariants(i1s, i2s, js, i4s_2, i5s_2)
    W2_no_fiber = mat2.sef_from_invariants(i1s, i2s, js)  # isotropic + volumetric only
    W2_fiber_only = W2_full - W2_no_fiber

    W_total_expr = W1_expr + W2_fiber_only

    param_syms = list(mat1._symbols.values())
    all_syms = (i1s, i2s, js, i4s_1, i5s_1, i4s_2, i5s_2, *param_syms)
    fn_total = lambdify(all_syms, W_total_expr, modules="numpy")
    fn_fiber1 = lambdify((i1s, i2s, js, i4s_1, i5s_1, *param_syms), W1_expr, modules="numpy")
    fn_fiber2 = lambdify((i1s, i2s, js, i4s_2, i5s_2, *param_syms), W2_fiber_only, modules="numpy")

    for lam in stretches:
        F = np.array([[[lam, 0, 0], [0, lam, 0], [0, 0, 1.0 / (lam * lam)]]])
        C = Kinematics.right_cauchy_green(F)

        # Compute invariants
        i1 = Kinematics.isochoric_invariant1(C)
        i2 = Kinematics.isochoric_invariant2(C)
        j = np.sqrt(Kinematics.det_invariant(C))
        i4_1 = Kinematics.fiber_invariant4(C, fiber1)
        i5_1 = Kinematics.fiber_invariant5(C, fiber1)
        i4_2 = Kinematics.fiber_invariant4(C, fiber2)
        i5_2 = Kinematics.fiber_invariant5(C, fiber2)

        pvals = list(mat1._params.values())
        W1 = float(fn_fiber1(i1[0], i2[0], j[0], i4_1[0], i5_1[0], *pvals))
        W2 = float(fn_fiber2(i1[0], i2[0], j[0], i4_2[0], i5_2[0], *pvals))
        Wt = float(fn_total(i1[0], i2[0], j[0], i4_1[0], i5_1[0], i4_2[0], i5_2[0], *pvals))

        print(f"{lam:8.3f} {W1:12.4f} {W2:12.4f} {Wt:12.4f}")


if __name__ == "__main__":
    main()
