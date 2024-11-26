from typing import Any, Iterable

from sympy import (
    Expr,
    ImmutableDenseNDimArray,
    Matrix,
    Symbol,
    log,
)

from hyper_surrogate.symbolic import SymbolicHandler


class Material(SymbolicHandler):
    """
    Material class for defining the constitutive model of the material.
    The class is inherited from the SymbolicHandler class and provides
    the necessary methods to define the constitutive model in symbolic form.

    Args:
        parameters: Iterable[Any] - The material parameters as a list of strings

    Properties:
        sef: The strain energy function in symbolic form

    Methods:
        pk2() -> Callable[..., Any]: Returns the second Piola-Kirchhoff stress tensor
        cmat() -> Callable[..., Any]: Returns the material stiffness tensor
    """

    def __init__(self, parameters: Iterable[str]) -> None:
        super().__init__()
        self.parameters = parameters

    @property
    def sef(self) -> Expr:
        """Strain energy function in symbolic form."""
        # Dummy placeholder
        return Symbol("sef")

    @property
    def pk2_symb(self) -> Matrix:
        """Second Piola-Kirchhoff stress tensor in symbolic form."""
        return self.pk2_tensor(self.sef)

    @property
    def cmat_symb(self) -> ImmutableDenseNDimArray:
        """Material stiffness tensor in symbolic form."""
        return self.cmat_tensor(self.pk2_symb)

    def sigma_symb(self, f: Matrix) -> Matrix:
        """Cauchy stress tensor in symbolic form."""
        return self.pushforward_2nd_order(self.pk2_symb, f)

    def smat_symb(self, f: Matrix) -> Matrix:
        """Material stiffness tensor in spatial form."""
        return self.pushforward_4th_order(self.cmat_symb, f)

    def jr_symb(self, f: Matrix) -> Matrix:
        """Jaumann rate contribution to the tangent tensor in symbolic form."""
        return self.jr(self.sigma_symb(f))

    def pk2(self) -> Any:
        """Second Piola-Kirchhoff stress tensor generator of numerical form."""
        return self.lambda_tensor(self.pk2_symb, *self.parameters)

    def cmat(self) -> Any:
        """Material stiffness tensor generator of numerical form."""
        return self.lambda_tensor(self.cmat_symb, *self.parameters)

    def sigma(self, f: Matrix) -> Any:
        """Cauchy stress tensor generator of numerical form."""
        return self.lambda_tensor(self.sigma_symb(f), *self.parameters)

    def smat(self, f: Matrix) -> Any:
        """Material stiffness tensor generator of numerical form."""
        return self.lambda_tensor(self.smat_symb(f), *self.parameters)

    # VOIGT NOTATION handlers
    def cauchy(self, f: Matrix) -> Matrix:
        """Reduce Cauchy stress tensor to 6x1 matrix using Voigt notation."""
        return self.reduce_2nd_order(self.sigma_symb(f))

    def tangent(self, f: Matrix, use_jaumann_rate: bool = False) -> Matrix:
        """Reduce tangent tensor to 6x6 matrix using Voigt notation."""
        tangent = self.smat_symb(f)
        if use_jaumann_rate:
            tangent += self.jr_symb(f)
        return self.reduce_4th_order(tangent)


class NeoHooke(Material):
    """
    Neo-Hookean material model for hyperelastic materials.
    The class inherits from the Material class and provides the necessary
    methods to define the Neo-Hookean model in symbolic form.

    Properties:
        sef: The strain energy function in symbolic form
    """

    def __init__(self) -> None:
        params = ["C10", "KBULK"]
        super().__init__(params)

    @property
    def sef(self) -> Expr:
        return (self.invariant1 - 3) * Symbol("C10") + 0.25 * Symbol("KBULK") * (
            self.invariant3 - 1 - log(self.invariant3)
        )


class MooneyRivlin(Material):
    """
    Mooney-Rivlin material model for hyperelastic materials.
    The class inherits from the Material class and provides the necessary
    methods to define the Mooney-Rivlin model in symbolic form.

    Properties:
        sef: The strain energy function in symbolic form
    """

    def __init__(self) -> None:
        params = ["C10", "C01"]
        super().__init__(params)

    @property
    def sef(self) -> Expr:
        return (self.invariant1 - 3) * Symbol("C10") + (self.invariant2 - 3) * Symbol("C01")
