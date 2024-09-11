from typing import Any, Iterable

import sympy as sym

from hyper_surrogate.symbolic import SymbolicHandler


class Material(SymbolicHandler):
    """
    Material class for defining the constitutive model of the material.
    The class is inherited from the SymbolicHandler class and provides
    the necessary methods to define the constitutive model in symbolic form.

    args:
    parameters: Iterable[Any] - The material parameters as a list of strings

    properties:
    sef: The strain energy function in symbolic form

    methods:
    pk2() -> Any: Returns the second Piola-Kirchhoff stress tensor
    cmat() -> Any: Returns the material stiffness tensor
    """

    def __init__(self, parameters: Iterable[Any]) -> None:
        super().__init__()
        self.parameters = parameters

    @property
    def sef(self) -> Any:
        """Strain energy function in symbolic form."""
        # Dummy placeholder
        return sym.Symbol("sef")

    @property
    def pk2_symb(self) -> Any:
        """Second Piola-Kirchhoff stress tensor in symbolic form."""
        return self.pk2_tensor(self.sef)

    @property
    def cmat_symb(self) -> Any:
        """Material stiffness tensor in symbolic form."""
        return self.cmat_tensor(self.pk2_symb)

    def sigma_symb(self, f: sym.Matrix) -> Any:
        """Cauchy stress tensor in symbolic form."""
        return self.pushforward_2nd_order(self.pk2_symb, f)

    def smat_symb(self, f: sym.Matrix) -> Any:
        """Material stiffness tensor in spatial form."""
        return self.pushforward_4th_order(self.cmat_symb, f)

    def cauchy(self, f: sym.Matrix) -> Any:
        return self.reduce_2nd_order(self.sigma_symb(f))

    def tangent(self, f: sym.Matrix) -> Any:
        # TODO:        # implement jaumman_rate_mat
        return self.reduce_4th_order(self.smat_symb(f))

    def pk2(self) -> Any:
        """Second Piola-Kirchhoff stress tensor generator of numerical form."""
        return self.lambdify(self.pk2_symb, *self.parameters)

    def cmat(self) -> Any:
        """Material stiffness tensor generator of numerical form."""
        return self.lambdify(self.cmat_symb, *self.parameters)

    def sigma(self, f: sym.Matrix) -> Any:
        """Cauchy stress tensor generator of numerical form."""
        return self.lambdify(self.sigma_symb(f), *self.parameters)

    def smat(self, f: sym.Matrix) -> Any:
        """Material stiffness tensor generator of numerical form."""
        return self.lambdify(self.smat_symb(f), *self.parameters)


class NeoHooke(Material):
    """
    Neo-Hookean material model for hyperelastic materials.
    The class inherits from the Material class and provides the necessary
    methods to define the Neo-Hookean model in symbolic form.

    properties:
    sef: The strain energy function in symbolic form
    """

    def __init__(self) -> None:
        params = ["C10"]
        super().__init__(params)

    @property
    def sef(self) -> Any:
        return (self.invariant1 - 3) * sym.Symbol("C10")


class MooneyRivlin(Material):
    """
    Mooney-Rivlin material model for hyperelastic materials.
    The class inherits from the Material class and provides the necessary
    methods to define the Mooney-Rivlin model in symbolic form.

    properties:
    sef: The strain energy function in symbolic form
    """

    def __init__(self) -> None:
        params = ["C10", "C01"]
        super().__init__(params)

    @property
    def sef(self) -> Any:
        return (self.invariant1 - 3) * sym.Symbol("C10") + (self.invariant2 - 3) * sym.Symbol("C01")
