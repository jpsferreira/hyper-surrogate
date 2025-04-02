import logging
from typing import Any, Dict, Iterable

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
    Base class for defining constitutive material models.
    The class is inherited from the SymbolicHandler class and provides
    the necessary methods to define the constitutive model in symbolic form.

    Args:
        parameters (Iterable[str]): Parameter names for the material model.

    Properties:
        sef: The strain energy function in symbolic form

    Methods:
        pk2() -> Callable[..., Any]: Returns the second Piola-Kirchhoff stress tensor
        cmat() -> Callable[..., Any]: Returns the material stiffness tensor
        get_default_parameters: Returns default parameter values for the material
        validate_parameters: Validates provided parameter values
    """

    def __init__(self, parameters: Iterable[str]) -> None:
        super().__init__()
        self.parameters = list(parameters)

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

    def get_default_parameters(self) -> Dict[str, float]:
        """
        Get default parameter values for the material.

        Returns:
            Dict[str, float]: Dictionary of parameter names and their default values.
        """
        # Base implementation returns 1.0 for each parameter
        return {param: 1.0 for param in self.parameters}

    def validate_parameters(self, params: Dict[str, float]) -> Dict[str, float]:
        """
        Validate the provided parameter values. If parameters are missing,
        use default values. Raises error for unknown parameters.

        Args:
            params (Dict[str, float]): Dictionary of parameter names and values.

        Returns:
            Dict[str, float]: Dictionary of validated parameter values.

        Raises:
            ValueError: If unknown parameters are provided.
        """
        # Check for unknown parameters
        unknown_params = set(params.keys()) - set(self.parameters)
        if unknown_params:
            raise ValueError(unknown_params)

        # Use default values for missing parameters
        defaults = self.get_default_parameters()
        validated_params = defaults.copy()
        validated_params.update(params)

        logging.debug(f"Using parameters: {validated_params}")
        return validated_params


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
            self.invariant3 - 1 - 2 * log(self.invariant3**0.5)
        )

    def get_default_parameters(self) -> Dict[str, float]:
        """
        Get default parameter values for Neo-Hookean model.

        Returns:
            Dict[str, float]: Dictionary of parameter names and their default values.
        """
        return {
            "C10": 0.5,  # Default shear modulus (MPa)
            "KBULK": 1000.0,  # Default bulk modulus (MPa)
        }


class MooneyRivlin(Material):
    """
    Mooney-Rivlin material model for hyperelastic materials.
    The class inherits from the Material class and provides the necessary
    methods to define the Mooney-Rivlin model in symbolic form.

    Properties:
        sef: The strain energy function in symbolic form
    """

    def __init__(self) -> None:
        params = ["C10", "C01", "KBULK"]
        super().__init__(params)

    @property
    def sef(self) -> Expr:
        return (
            (self.invariant1 - 3) * Symbol("C10")
            + (self.invariant2 - 3) * Symbol("C01")
            + 0.25 * Symbol("KBULK") * (self.invariant3 - 1 - 2 * log(self.invariant3**0.5))
        )

    def get_default_parameters(self) -> Dict[str, float]:
        """
        Get default parameter values for Mooney-Rivlin model.

        Returns:
            Dict[str, float]: Dictionary of parameter names and their default values.
        """
        return {
            "C10": 0.3,  # Default first shear modulus (MPa)
            "C01": 0.2,  # Default second shear modulus (MPa)
            "KBULK": 1000.0,  # Default bulk modulus (MPa)
        }
