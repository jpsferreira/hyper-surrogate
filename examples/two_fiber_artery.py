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

    for lam in stretches:
        F = np.array([[[lam, 0, 0], [0, lam, 0], [0, 0, 1.0 / (lam * lam)]]])
        C = Kinematics.right_cauchy_green(F)

        # Compute invariants for each fiber family
        i1 = Kinematics.isochoric_invariant1(C)
        j = np.sqrt(Kinematics.det_invariant(C))
        i4_1 = Kinematics.fiber_invariant4(C, fiber1)
        i4_2 = Kinematics.fiber_invariant4(C, fiber2)

        # Total energy needs proper evaluation
        from sympy import Symbol, lambdify

        syms = [Symbol("I1b"), Symbol("I2b"), Symbol("J"), Symbol("I4"), Symbol("I5")]
        W1_expr = mat1.sef_from_invariants(*syms)
        param_syms1 = list(mat1._symbols.values())
        fn1 = lambdify((*syms, *param_syms1), W1_expr, modules="numpy")

        i2 = Kinematics.isochoric_invariant2(C)
        i5_1 = Kinematics.fiber_invariant5(C, fiber1)
        i5_2 = Kinematics.fiber_invariant5(C, fiber2)

        W1 = float(fn1(i1[0], i2[0], j[0], i4_1[0], i5_1[0], *mat1._params.values()))

        W2_expr = mat2.sef_from_invariants(*syms)
        param_syms2 = list(mat2._symbols.values())
        fn2 = lambdify((*syms, *param_syms2), W2_expr, modules="numpy")
        W2 = float(fn2(i1[0], i2[0], j[0], i4_2[0], i5_2[0], *mat2._params.values()))

        print(f"{lam:8.3f} {W1:12.4f} {W2:12.4f} {W1 + W2:12.4f}")


if __name__ == "__main__":
    main()
