"""Define a custom strain-energy function and emit a Fortran UMAT.

Path B of the hyper-surrogate pipeline: when your hyperelastic model is
not in the built-in catalogue but you can write it down as a closed-form
expression in the standard invariants, subclass ``Material`` with a
``sef`` property and hand the instance to ``UMATHandler`` -- no neural
network, no training, no Fortran written by hand.

Usage:
    uv run python examples/custom_sef.py
"""

from __future__ import annotations

from typing import ClassVar

import sympy as sp

from hyper_surrogate import Material, UMATHandler


class MyOgdenLike(Material):
    """One-term Ogden-style isochoric SEF plus a quadratic volumetric term.

    W(I1bar, J) = (mu / alpha) * (I1bar^(alpha/2) - 3)  +  (K / 2) * (J - 1)^2
    """

    DEFAULT_PARAMS: ClassVar[dict[str, float]] = {"mu": 1.0, "alpha": 2.0, "KBULK": 1000.0}

    def __init__(self, parameters: dict[str, float] | None = None) -> None:
        super().__init__({**self.DEFAULT_PARAMS, **(parameters or {})})

    @property
    def sef(self) -> sp.Expr:
        h = self._handler
        mu, alpha, K = (self._symbols[name] for name in ("mu", "alpha", "KBULK"))
        return (mu / alpha) * (h.isochoric_invariant1 ** (alpha / 2) - 3) + 0.5 * K * (sp.sqrt(h.invariant3) - 1) ** 2


if __name__ == "__main__":
    print("── 1. Defining custom SEF (MyOgdenLike) ──")
    material = MyOgdenLike()
    print(f"  Parameters: {material._params}")

    print("\n── 2. Generating analytical UMAT ──")
    UMATHandler(material).generate("mysef.f90")
    print("  Output: mysef.f90")
    print("\nThis file is a complete Abaqus-compatible UMAT, generated")
    print("from a single symbolic expression -- no training required.")
