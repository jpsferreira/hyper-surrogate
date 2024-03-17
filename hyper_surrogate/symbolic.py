from typing import Any

import numpy as np
import sympy as sym


class SymbolicHandler:
    """
    A class that handles symbolic computations using SymPy.

    Attributes:
        c_tensor (sym.Matrix): A 3x3 matrix of symbols.
    """

    def __init__(self) -> None:
        self.c_tensor = self._c_tensor()

    def _c_tensor(self) -> sym.Matrix:
        """
        Create a 3x3 matrix of symbols.

        Returns:
            sym.Matrix: A 3x3 matrix of symbols.
        """
        return sym.Matrix(3, 3, lambda i, j: sym.Symbol(f"C_{i+1}{j+1}"))

    @property
    def invariant1(self) -> Any:
        """
        Compute the first invariant of the c_tensor.

        Returns:
            sym.Expr: The first invariant of the c_tensor.
        """
        I3 = self.invariant3  # Determinant
        trace = self.c_tensor.trace()  # Trace
        return trace * (I3 ** (-sym.Rational(1, 3)))

    @property
    def invariant3(self) -> Any:
        """
        Compute the third invariant of the c_tensor.

        Returns:
            sym.Expr: The third invariant of the c_tensor.
        """
        return self.c_tensor.det()

    def pk2_tensor(self, sef: sym.Expr) -> sym.MutableDenseNDimArray:
        """
        Compute the pk2 tensor.

        Args:
            sef (sym.Expr): The strain energy function.

        Returns:
            sym.Matrix: The pk2 tensor.
        """
        return sym.MutableDenseNDimArray([[sym.diff(sef, self.c_tensor[i, j]) for j in range(3)] for i in range(3)])

    def cmat_tensor(self, pk2: sym.Matrix) -> sym.MutableDenseNDimArray:
        """
        Compute the cmat tensor.

        Args:
            pk2 (sym.Matrix): The pk2 tensor.

        Returns:
            sym.MutableDenseNDimArray: The cmat tensor.
        """
        return sym.MutableDenseNDimArray(
            [
                [[[sym.diff(pk2[i, j], self.c_tensor[k, ll]) for ll in range(3)] for k in range(3)] for j in range(3)]
                for i in range(3)
            ]
        )

    def substitute(self, symbolic_tensor: sym.Expr, numerical_c_tensor: np.ndarray) -> Any:
        """
        Automatically substitute numerical values from a given 3x3 numerical matrix into c_tensor.

        Args:
            symbolic_tensor (sym.Matrix): A symbolic tensor to substitute numerical values into.
            numerical_matrix (np.ndarray): A 3x3 numerical matrix to substitute into c_tensor.

        Returns:
            sym.Matrix: The c_tensor with numerical values substituted.

        Raises:
            ValueError: If numerical_tensor is not a 3x3 matrix.
        """
        return symbolic_tensor.subs(
            {self.c_tensor[i, j]: np.array(numerical_c_tensor)[i, j] for i in range(3) for j in range(3)}
        )
