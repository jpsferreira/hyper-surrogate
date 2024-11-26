from typing import Any, Generator, Iterable, List

import numpy as np
from sympy import (
    Expr,
    ImmutableDenseNDimArray,
    ImmutableMatrix,
    Matrix,
    MutableDenseMatrix,
    Rational,
    Symbol,
    diff,
    lambdify,
    simplify,
)


class SymbolicHandler:
    """
    A class that handles symbolic computations using SymPy.

    Attributes:
        c_tensor (Matrix): A 3x3 matrix of symbols.
    """

    def __init__(self) -> None:
        self.c_tensor = self._c_tensor()
        self.f_tensor = self._f_tensor()

    def f_symbols(self) -> List[Symbol]:
        """
        Return the f_tensor flattened symbols.

        Returns:
            list: A list of f_tensor symbols.
        """
        return [self.f_tensor[i, j] for i in range(3) for j in range(3)]

    def _f_tensor(self) -> ImmutableMatrix:
        """
        Create a 3x3 matrix of symbols.

        Returns:
            Matrix: A 3x3 matrix of symbols.
        """
        return ImmutableMatrix(3, 3, lambda i, j: Symbol(f"F_{i + 1}{j + 1}"))

    def c_symbols(self) -> List[Symbol]:
        """
        Return the c_tensor flattened symbols.

        Returns:
            list: A list of c_tensor symbols.
        """
        return [self.c_tensor[i, j] for i in range(3) for j in range(3)]

    def _c_tensor(self) -> ImmutableMatrix:
        """
        Create a 3x3 matrix of symbols.

        Returns:
            Matrix: A 3x3 matrix of symbols.
        """
        return ImmutableMatrix(3, 3, lambda i, j: Symbol(f"C_{i + 1}{j + 1}"))

    # multuply c_tensor by itself
    def _c_tensor_squared(self) -> ImmutableMatrix:
        """
        Compute the square of the c_tensor.

        Returns:
            Matrix: The square of the c_tensor.
        """
        # matrix product
        return self.c_tensor.multiply(self.c_tensor)

    @property
    def invariant1(self) -> Expr:
        """
        Compute the first invariant of the c_tensor.

        Returns:
            Expr: The first invariant of the c_tensor.
        """
        I3 = self.invariant3  # Determinant
        trace = self.c_tensor.trace()  # Trace
        return trace * (I3 ** (-Rational(1, 3)))

    @property
    def invariant2(self) -> Expr:
        """
        Compute the second invariant of the c_tensor.

        Returns:
            Expr: The second invariant of the c_tensor.
        """
        c_squared = self._c_tensor_squared()
        return Rational(1, 2) * (self.c_tensor.multiply(self.c_tensor).trace() - c_squared.trace())

    @property
    def invariant3(self) -> Expr:
        """
        Compute the third invariant of the c_tensor.

        Returns:
            Expr: The third invariant of the c_tensor.
        """
        return self.c_tensor.det()

    def pk2_tensor(self, sef: Expr) -> Matrix:
        """
        Compute the pk2 tensor.

        Args:
            sef (Expr): The strain energy function.

        Returns:
            Matrix: The pk2 tensor.
        """
        pk2 = Matrix([[diff(sef, self.c_tensor[i, j]) for j in range(3)] for i in range(3)])
        # force symmetry
        return pk2 + pk2.T

    def cmat_tensor(self, pk2: Matrix) -> ImmutableDenseNDimArray:
        """
        Compute the cmat tensor.

        Args:
            pk2 (Matrix): The pk2 tensor.

        Returns:
            ImmutableDenseNDimArray: The stiffness tensor (3x3x3x3) with minor symmetry.
        """
        return ImmutableDenseNDimArray([
            [
                [
                    [(diff(pk2[i, j], self.c_tensor[k, ll]) + diff(pk2[i, j], self.c_tensor[ll, k])) for ll in range(3)]
                    for k in range(3)
                ]
                for j in range(3)
            ]
            for i in range(3)
        ])

    def sigma_tensor(self, sef: Expr, f: Matrix) -> Matrix:
        """
        Compute the sigma tensor.

        Args:
            sef (Expr): The strain energy function.
            f (Matrix): The deformation gradient tensor.

        Returns:
            Matrix: The Cauchy stress tensor.
        """
        return self.pushforward_2nd_order(self.pk2_tensor(sef), f)

    def smat_tensor(self, pk2: Matrix, f: Matrix) -> ImmutableDenseNDimArray:
        """
        Compute the material stiffness tensor.

        Args:
            pk2 (Matrix): The pk2 tensor.
            f (Matrix): The deformation gradient tensor.

        Returns:
            ImmutableDenseNDimArray: The material stiffness tensor.
        """
        return self.pushforward_4th_order(self.cmat_tensor(pk2), f)

    def substitute(
        self,
        symbolic_tensor: MutableDenseMatrix,
        numerical_c_tensor: np.ndarray,
        *args: dict,
    ) -> Matrix:
        """
        Automatically substitute numerical values from a given 3x3 numerical matrix into c_tensor.

        Args:
            symbolic_tensor (Matrix): A symbolic tensor to substitute numerical values into.
            numerical_c_tensor (np.ndarray): A 3x3 numerical matrix to substitute into c_tensor.
            args (dict): Additional substitution dictionaries.

        Returns:
            Matrix: The symbolic_tensor with numerical values substituted.

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
        symbolic_tensor: MutableDenseMatrix,
        numerical_c_tensors: np.ndarray,
        *args: dict,
    ) -> Generator[Matrix, None, None]:
        """
        Automatically substitute numerical values from a given 3x3 numerical matrix into c_tensor.

        Args:
            symbolic_tensor (Matrix): A symbolic tensor to substitute numerical values into.
            numerical_c_tensors (np.ndarray): N 3x3 numerical matrices to substitute into c_tensor.
            args (dict): Additional substitution dictionaries.

        Returns:
            Generator[Matrix, None, None]: The symbolic_tensor with numerical values substituted.

        Raises:
            ValueError: If numerical_tensor is not a 3x3 matrix.
        """
        for numerical_c_tensor in numerical_c_tensors:
            yield self.substitute(symbolic_tensor, numerical_c_tensor, *args)

    def lambda_tensor(self, symbolic_tensor: Matrix, *args: Iterable[Any]) -> Any:
        """
        Create a lambdified function from a symbolic tensor that can be used for numerical evaluation.

        Args:
            symbolic_tensor (Expr or Matrix): The symbolic tensor to be lambdified.
            args (dict): Additional substitution lists of symbols.
        Returns:
            function: A function that can be used to numerically evaluate the tensor with specific values.
        """

        return lambdify((self.c_symbols(), *args), symbolic_tensor.tolist(), modules="numpy")

    def _evaluate(self, lambdified_tensor: Any, *args: Any, **kwargs: Any) -> Any:
        """
        Evaluate a lambdified tensor with specific values.

        Args:
            lambdified_tensor (function): A lambdified tensor function.
            args (dict): Additional substitution lists of symbols.
            kwargs (dict): Additional keyword arguments.

        Returns:
            Generator[Any, None, None]: The evaluated tensor.
        """
        return lambdified_tensor(*args, **kwargs)

    def evaluate_iterator(
        self, lambdified_tensor: Any, numerical_c_tensors: np.ndarray, *args: Any, **kwargs: Any
    ) -> Generator[Any, None, None]:
        """
        Evaluate a lambdified tensor with specific values.

        Args:
            lambdified_tensor (function): A lambdified tensor function.
            args (dict): Additional substitution lists of symbols.
            kwargs (dict): Additional keyword arguments.

        Returns:
            Generator[Any, None, None]: The evaluated tensor.
        """
        for numerical_c_tensor in numerical_c_tensors:
            yield self._evaluate(lambdified_tensor, numerical_c_tensor.flatten(), *args, **kwargs)

    @staticmethod
    def reduce_2nd_order(tensor: Matrix) -> Matrix:
        """
        Convert a 3x3 matrix to 6x1 matrix using Voigt notation

        Args:
            tensor (sp.Matrix): A 3x3 symmetric matrix.

        Returns:
            sp.Matrix: A 6x1 matrix.
        """
        # Validate the input tensor dimensions
        if tensor.shape != (3, 3):
            raise ValueError("Wrong.shape.")
        # Voigt notation conversion: xx, yy, zz, xy, xz, yz
        return Matrix([
            tensor[0, 0],  # xx
            tensor[1, 1],  # yy
            tensor[2, 2],  # zz
            tensor[0, 1],  # xy
            tensor[0, 2],  # xz
            tensor[1, 2],  # yz
        ])

    @staticmethod
    def reduce_4th_order(tensor: ImmutableDenseNDimArray) -> Matrix:
        """
        Convert a 3x3x3x3 matrix to 6x6 matrix using Voigt notation

        Args:
            tensor (ImmutableDenseNDimArray): A 3x3x3x3 matrix.

        Returns:
            Matrix: A 6x6 matrix.
        """
        # Validate the input tensor dimensions
        if tensor.shape != (3, 3, 3, 3):
            raise ValueError("Wrong.shape.")

        # Voigt notation index mapping
        ii1 = [0, 1, 2, 0, 0, 1]
        ii2 = [0, 1, 2, 1, 2, 2]

        # Initialize a 6x6 matrix
        voigt_matrix = Matrix.zeros(6, 6)

        # Fill the Voigt matrix by averaging symmetric components of the 4th-order tensor
        for i in range(6):
            for j in range(6):
                # Access the relevant tensor components using ii1 and ii2 mappings
                pp1 = tensor[ii1[i], ii2[i], ii1[j], ii2[j]]
                pp2 = tensor[ii1[i], ii2[i], ii2[j], ii1[j]]
                # Average the two permutations to ensure symmetry
                voigt_matrix[i, j] = 0.5 * (pp1 + pp2)
        return voigt_matrix

    @staticmethod
    def pushforward_2nd_order(tensor2: Matrix, f: Matrix) -> Matrix:
        """
        Push forward a 2nd order tensor in material configuration.

        args:
        tensor2: Any - The 2nd order tensor
        f: Any - The deformation gradient tensor

        returns:
        Any - The pushforwarded 2nd order tensor
        """
        return simplify(f * tensor2 * f.T)

    @staticmethod
    def pushforward_4th_order(tensor4: ImmutableDenseNDimArray, f: Matrix) -> ImmutableDenseNDimArray:
        """
        Push forward a 4th order tensor in material configuration.

        Args:
            tensor4 (MutableDenseNDimArray): The 4th order tensor.
            f (Matrix): The deformation gradient tensor (2nd order tensor).

        Returns:
            ImmutableDenseNDimArray: The pushforwarded 4th order tensor.
        """
        # Calculate the pushforwarded tensor using comprehensions and broadcasting
        return ImmutableDenseNDimArray([
            [
                [
                    [
                        sum(
                            f[i, ii] * f[j, jj] * f[k, kk] * f[l0, ll] * tensor4[ii, jj, kk, ll]
                            for ii in range(3)
                            for jj in range(3)
                            for kk in range(3)
                            for ll in range(3)
                        )
                        for l0 in range(3)
                    ]
                    for k in range(3)
                ]
                for j in range(3)
            ]
            for i in range(3)
        ])

    @staticmethod
    def jr(sigma: Matrix) -> ImmutableDenseNDimArray:
        """
        Compute the Jaumann rate contribution for the spatial elasticity tensor.

        Args:
            sigma (Matrix): The Cauchy stress tensor (2nd order tensor).

        Returns:
            ImmutableDenseNDimArray: The Jaumann rate contribution (4th order tensor).
        """
        # Ensure sigma is a 3x3 matrix
        if sigma.shape != (3, 3):
            raise ValueError("Wrongshape")

        # Use list comprehension to build the 4th-order tensor directly
        return ImmutableDenseNDimArray([
            [
                [
                    [
                        (1 / 2)
                        * (
                            int(i == k) * sigma[j, ll]
                            + sigma[i, k] * int(j == ll)
                            + int(i == ll) * sigma[j, k]
                            + sigma[i, ll] * int(j == k)
                        )
                        for ll in range(3)
                    ]
                    for k in range(3)
                ]
                for j in range(3)
            ]
            for i in range(3)
        ])
