from typing import Any, Iterable

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

    def c_symbols(self) -> Any:
        """
        Return the c_tensor flattened symbols.

        Returns:
            list: A list of c_tensor symbols.
        """
        return [self.c_tensor[i, j] for i in range(3) for j in range(3)]

    def _c_tensor(self) -> sym.Matrix:
        """
        Create a 3x3 matrix of symbols.

        Returns:
            sym.Matrix: A 3x3 matrix of symbols.
        """
        return sym.Matrix(3, 3, lambda i, j: sym.Symbol(f"C_{i+1}{j+1}"))

    # multuply c_tensor by itself
    def _c_tensor_squared(self) -> Any:
        """
        Compute the square of the c_tensor.

        Returns:
            sym.Matrix: The square of the c_tensor.
        """
        # matrix product
        return self.c_tensor.multiply(self.c_tensor)

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
    def invariant2(self) -> Any:
        """
        Compute the second invariant of the c_tensor.

        Returns:
            sym.Expr: The second invariant of the c_tensor.
        """
        c_squared = self._c_tensor_squared()
        return sym.Rational(1, 2) * (self.c_tensor.multiply(self.c_tensor).trace() - c_squared.trace())

    @property
    def invariant3(self) -> Any:
        """
        Compute the third invariant of the c_tensor.

        Returns:
            sym.Expr: The third invariant of the c_tensor.
        """
        return self.c_tensor.det()

    def pk2_tensor(self, sef: sym.Expr) -> sym.Matrix:
        """
        Compute the pk2 tensor.

        Args:
            sef (sym.Expr): The strain energy function.

        Returns:
            sym.Matrix: The pk2 tensor.
        """
        return sym.Matrix([[sym.diff(sef, self.c_tensor[i, j]) for j in range(3)] for i in range(3)])

    def cmat_tensor(self, pk2: sym.Matrix) -> sym.ImmutableDenseNDimArray:
        """
        Compute the cmat tensor.

        Args:
            pk2 (sym.Matrix): The pk2 tensor.

        Returns:
            sym.MutableDenseNDimArray: The cmat tensor.
        """
        return sym.ImmutableDenseNDimArray(
            [
                [[[sym.diff(pk2[i, j], self.c_tensor[k, ll]) for ll in range(3)] for k in range(3)] for j in range(3)]
                for i in range(3)
            ]
        )

    def substitute(
        self,
        symbolic_tensor: sym.MutableDenseMatrix,
        numerical_c_tensor: np.ndarray,
        *args: dict,
    ) -> Any:
        """
        Automatically substitute numerical values from a given 3x3 numerical matrix into c_tensor.

        Args:
            symbolic_tensor (sym.Matrix): A symbolic tensor to substitute numerical values into.
            numerical_matrix (np.ndarray): A 3x3 numerical matrix to substitute into c_tensor.
            args (dict): Additional substitution dictionaries.

        Returns:
            sym.Matrix: The symbolic_tensor with numerical values substituted.

        Raises:
            ValueError: If numerical_tensor is not a 3x3 matrix.
        """
        if not isinstance(numerical_c_tensor, np.ndarray) or numerical_c_tensor.shape != (3, 3):
            raise ValueError("c_tensor.shape")

        # Start with substitutions for c_tensor elements
        substitutions = {self.c_tensor[i, j]: numerical_c_tensor[i, j] for i in range(3) for j in range(3)}
        # Merge additional substitution dictionaries from *args
        substitutions.update(*args)
        return symbolic_tensor.subs(substitutions)

    def substitute_iterator(
        self,
        symbolic_tensor: sym.MutableDenseMatrix,
        numerical_c_tensors: np.ndarray,
        *args: dict,
    ) -> Any:
        """
        Automatically substitute numerical values from a given 3x3 numerical matrix into c_tensor.

        Args:
            symbolic_tensor (sym.Matrix): A symbolic tensor to substitute numerical values into.
            numerical_matrix (np.ndarray): A 3x3 numerical matrix to substitute into c_tensor.
            args (dict): Additional substitution dictionaries.

        Returns:
            sym.Matrix: The symbolic_tensor with numerical values substituted.

        Raises:
            ValueError: If numerical_tensor is not a 3x3 matrix.
        """
        for numerical_c_tensor in numerical_c_tensors:
            yield self.substitute(symbolic_tensor, numerical_c_tensor, *args)

    def lambdify(self, symbolic_tensor: sym.Matrix, *args: Iterable[Any]) -> Any:
        """
        Create a lambdified function from a symbolic tensor that can be used for numerical evaluation.

        Args:
            tensor (sym.Expr or sym.Matrix): The symbolic tensor to be lambdified.
            args (dict): Additional substitution lists of symbols.
        Returns:
            function: A function that can be used to numerically evaluate the tensor with specific values.
        """

        return sym.lambdify((self.c_symbols(), *args), symbolic_tensor, modules="numpy")

    def _evaluate(self, lambdified_tensor: Any, *args: Any) -> Any:
        """
        Evaluate a lambdified tensor with specific values.

        Args:
            lambdified_tensor (function): A lambdified tensor function.
            args (dict): Additional substitution lists of symbols.

        Returns:
            Any: The evaluated tensor.
        """
        return lambdified_tensor(*args)

    def evaluate_iterator(self, lambdified_tensor: Any, numerical_c_tensors: np.ndarray, *args: Any) -> Any:
        """
        Evaluate a lambdified tensor with specific values.

        Args:
            lambdified_tensor (function): A lambdified tensor function.
            args (dict): Additional substitution lists of symbols.

        Returns:
            Any: The evaluated tensor.
        """
        for numerical_c_tensor in numerical_c_tensors:
            yield self._evaluate(lambdified_tensor, numerical_c_tensor.flatten(), *args)
